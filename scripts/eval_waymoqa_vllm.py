#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WaymoQA inference script for vLLM (OpenAI-compatible Chat Completions).

- ImageQA (type=image):
    Loads 8 camera images:
      {token}_{frame:03d}_{CAM}.jpg
    in a fixed 8-view order, and sends them as multi-image inputs.

- VideoQA (type=video):
    Does NOT use mp4 files.
    Instead, loads mosaic frames from a directory:
      {token}_{idx:03d}.jpg (or token_{idx}_suffix.jpg)
    Samples frames by stride and sends them as an ordered image sequence.
    The prompt explicitly explains the mosaic layout (view directions).

Outputs:
- CSV:  <save_dir>/pred_<model>_<split>.csv
- TXT:  <save_dir>/summary_<model>_<split>.txt
"""

import os
import re
import json
import base64
import argparse
import time
import threading
import csv
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# ----------------------------
# Defaults (edit for your repo)
# ----------------------------
DEFAULT_JSONL = "./dataset/questions/test.jsonl"

# ImageQA 8-view images live here: token_XXX_CAM.jpg
DEFAULT_IMG_ROOT = "./dataset/waymoe2e/test_imgs"

# VideoQA mosaics live here: token_XXX.jpg (multiple per token)
DEFAULT_MOSAIC_ROOT = "./dataset/waymoe2e/test_imgs"


# ----------------------------
# Constants / knobs
# ----------------------------
CHOICE_LETTERS = "ABCD"

MAX_SIDE = int(os.getenv("MAX_SIDE", "1024"))
JPEG_Q = int(os.getenv("JPEG_Q", "85"))

STRICT_SUFFIX = (
    "Respond with EXACTLY ONE LETTER from {A,B,C,D}. "
    "No words, no punctuation, no explanation."
)

# 6-view camera order (matches your exported filenames)
CAM_ORDER_8 = [
    "FRONT_LEFT",
    "FRONT",
    "FRONT_RIGHT",
    "SIDE_RIGHT",
    "SIDE_LEFT",
    "REAR_LEFT",
    "REAR",
    "REAR_RIGHT",
]

# Mosaic layout text (matches your mosaic exporter)
MOSAIC_LAYOUT_TEXT = (
    "Each frame is a 3x3 multi-view mosaic with a fixed layout:\n"
    "Top row:    FRONT LEFT | FRONT | FRONT RIGHT\n"
    "Middle row: SIDE LEFT  | (blank) | SIDE RIGHT\n"
    "Bottom row: REAR LEFT  | REAR  | REAR RIGHT\n"
)


# ----------------------------
# Thread-local OpenAI client
# ----------------------------
_client_local = threading.local()

def get_client(api_base: str, api_key: str) -> OpenAI:
    """
    OpenAI SDK client pointing to an OpenAI-compatible endpoint (e.g., vLLM).
    """
    c = getattr(_client_local, "c", None)
    if c is None:
        c = OpenAI(base_url=api_base, api_key=api_key)
        _client_local.c = c
    return c


# ----------------------------
# Utilities
# ----------------------------
def normalize_options(options: List[str]) -> List[str]:
    """
    Remove leading 'A. ' / 'B. ' ... prefixes if present.
    """
    out = []
    for s in options:
        s = re.sub(r'^[A-Da-d]\.\s*', '', str(s)).strip()
        out.append(s)
    return out


def resize_and_b64_img(img_bgr: np.ndarray, max_side: int = MAX_SIDE, q: int = JPEG_Q) -> str:
    """
    Resize by longest side and encode as base64 JPEG.
    """
    h, w = img_bgr.shape[:2]
    s = max(h, w) / float(max_side)
    if s > 1:
        img_bgr = cv2.resize(img_bgr, (int(w / s), int(h / s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def b64_image_resized(path: Path) -> str:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(str(path))
    return resize_and_b64_img(img)


def pad3(n: int) -> str:
    return str(int(n)).zfill(3)


def parse_choice(text: str, k: int) -> Optional[int]:
    """
    Parse a single-letter answer from model output.
    """
    t = re.sub(r"[^A-Z0-9]", "", text.strip().upper())
    if t:
        ch = t[0]
        if ch in CHOICE_LETTERS:
            idx = CHOICE_LETTERS.index(ch)
            return idx if idx < k else None
        if ch in "1234":
            idx = int(ch) - 1
            return idx if idx < k else None

    m = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    if m:
        return CHOICE_LETTERS.index(m.group(1).upper())

    m = re.search(r"\b([1-4])\b", text)
    if m:
        return int(m.group(1)) - 1

    return None


# ----------------------------
# CSV append
# ----------------------------
def csv_append_row(csv_path: Path, header: List[str], row: Dict, lock: threading.Lock):
    line = [row.get(h, "") for h in header]
    with lock:
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(header)
            w.writerow(line)


# ----------------------------
# Media caches (avoid repeated disk reads)
# ----------------------------
_media_cache_lock = threading.Lock()

# (token, frame_index) -> (payloads_b64, paths_joined)
_image6_cache: Dict[Tuple[str, int], Tuple[List[str], str]] = {}

# (token, stride, max_frames) -> (payloads_b64, paths_joined)
_mosaic_cache: Dict[Tuple[str, int, int], Tuple[List[str], str]] = {}


def get_eight_image_payloads(img_root: Path, token: str, frame_index: Optional[int]) -> Tuple[List[str], List[str], str]:
    """
    Load 8 camera images:
      {token}_{pad3(frame)}_{CAM}.jpg
    in the fixed order CAM_ORDER_8.

    Returns: (payloads_b64, paths, error_msg)
    """
    if frame_index is None:
        return [], [], "missing frame_index for image sample"

    key = (token, int(frame_index))
    with _media_cache_lock:
        if key in _image6_cache:
            payloads, paths_join = _image6_cache[key]
            return payloads, (paths_join.split(";") if paths_join else []), ""

    fidx_str = pad3(int(frame_index))

    paths: List[str] = []
    for cam in CAM_ORDER_8:
        p = img_root / f"{token}_{fidx_str}_{cam}.jpg"
        if not p.exists():
            return [], [], f"missing image: {p}"
        paths.append(str(p))

    payloads = [b64_image_resized(Path(p)) for p in paths]

    with _media_cache_lock:
        _image6_cache[key] = (payloads, ";".join(paths))
    return payloads, paths, ""


def _parse_mosaic_idx(token: str, filename: str) -> Optional[int]:
    """
    Accept both:
      token_010.jpg
      token_010_anything.jpg
    """
    m = re.match(rf"^{re.escape(token)}_(\d+).*\.jpg$", filename, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def get_mosaic_payloads(mosaic_root: Path, token: str, stride: int, max_frames: int) -> Tuple[List[str], List[str], str]:
    """
    Load mosaic frames from:
      {mosaic_root}/{token}_*.jpg
    Sort by index, then sample by stride.

    Returns: (payloads_b64, paths, error_msg)
    """
    stride = max(1, int(stride))
    max_frames = int(max_frames)

    key = (token, stride, max_frames)
    with _media_cache_lock:
        if key in _mosaic_cache:
            payloads, paths_join = _mosaic_cache[key]
            return payloads, (paths_join.split(";") if paths_join else []), ""

    cand = sorted(mosaic_root.glob(f"{token}_*.jpg"))
    if not cand:
        return [], [], f"no mosaic frames found for token={token} in {mosaic_root}"

    pairs = []
    for p in cand:
        idx = _parse_mosaic_idx(token, p.name)
        if idx is not None:
            pairs.append((idx, p))
    if not pairs:
        return [], [], f"found mosaic candidates but failed to parse indices for token={token}"

    pairs.sort(key=lambda x: x[0])
    sampled = pairs[0::stride]
    if max_frames > 0:
        sampled = sampled[:max_frames]

    paths = [str(p) for _, p in sampled]
    payloads = [b64_image_resized(Path(p)) for p in paths]

    with _media_cache_lock:
        _mosaic_cache[key] = (payloads, ";".join(paths))
    return payloads, paths, ""


# ----------------------------
# Chat Completions message builders
# ----------------------------
def build_messages_imageqa(question: str, options: List[str], payloads_b64: List[str]) -> List[dict]:
    options = normalize_options(options)
    opt_str = "\n".join([f"{CHOICE_LETTERS[i]}. {o}" for i, o in enumerate(options)])

    order_str = ", ".join([c.replace("_", " ") for c in CAM_ORDER_8])

    system = (
        "You are a driving VQA assistant.\n"
        "You will be given 8 camera views from the same timestamp.\n"
        f"The images are provided in this order: {order_str}.\n"
        "Use ALL views to answer.\n"
        + STRICT_SUFFIX
    )

    content = []
    for b64 in payloads_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    content.append({
        "type": "text",
        "text": f"Question:\n{question}\n\nOptions:\n{opt_str}\n\nAnswer with A/B/C/D only."
    })

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def build_messages_videoqa(question: str, options: List[str], frames_b64: List[str], stride: int) -> List[dict]:
    options = normalize_options(options)
    opt_str = "\n".join([f"{CHOICE_LETTERS[i]}. {o}" for i, o in enumerate(options)])

    system = (
        "You are a driving VQA assistant.\n"
        "You will be given a sequence of mosaic frames sampled from a driving clip.\n"
        "Inspect the frames in the given order.\n\n"
        + MOSAIC_LAYOUT_TEXT + "\n"
        f"(Frames are sampled by stride={stride}; adjust stride based on your GPU memory.)\n\n"
        + STRICT_SUFFIX
    )

    content = []
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    content.append({
        "type": "text",
        "text": f"Question:\n{question}\n\nOptions:\n{opt_str}\n\nAnswer with A/B/C/D only."
    })

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


# ----------------------------
# Model call (Chat Completions)
# ----------------------------
def call_model_chat(client: OpenAI, model_name: str, messages: List[dict],
                    max_tokens: int, temperature: float, top_p: float,
                    retries: int = 4) -> str:
    """
    Call vLLM OpenAI-compatible Chat Completions endpoint.
    """
    backoff = 1.5
    last = None
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last = e
            time.sleep(backoff ** i)
    raise RuntimeError(str(last) if last else "unknown error")


# ----------------------------
# One item
# ----------------------------
def process_one(item: Dict, cfg: Dict) -> Dict:
    client = get_client(cfg["api_base"], cfg["api_key"])

    token = item.get("token")
    typ = item.get("type")  # "image" | "video"
    q = item.get("question", "")
    cat = item.get("category", "Unknown")
    opts = item.get("options", [])
    ans = item.get("answer_index", None)
    frame_index = item.get("frame_index", None)

    if frame_index is not None:
        try:
            frame_index = int(frame_index)
        except Exception:
            frame_index = None

    if (token is None) or (typ not in ("image", "video")) or (not opts) or (ans is None):
        return {"skip": True}

    gt_idx = int(ans) - int(cfg["answer_index_base"])
    pred_idx: Optional[int] = None
    used_path = ""
    error = ""

    try:
        if typ == "image":
            payloads, paths, err = get_eight_image_payloads(cfg["img_root"], token, frame_index)
            used_path = ";".join(paths) if paths else ""
            if not payloads:
                error = err or "missing 8-view images"
            else:
                messages = build_messages_imageqa(q, opts, payloads)
                text = call_model_chat(
                    client=client,
                    model_name=cfg["model_name"],
                    messages=messages,
                    max_tokens=cfg["max_tokens"],
                    temperature=cfg["temperature"],
                    top_p=cfg["top_p"],
                )
                pred_idx = parse_choice(text, len(opts))

        else:
            frames_b64, paths, err = get_mosaic_payloads(
                cfg["mosaic_root"], token,
                stride=cfg["video_stride"],
                max_frames=cfg["video_max_frames"],
            )
            used_path = ";".join(paths) if paths else ""
            if not frames_b64:
                error = err or "missing mosaic frames"
            else:
                messages = build_messages_videoqa(q, opts, frames_b64, stride=cfg["video_stride"])
                text = call_model_chat(
                    client=client,
                    model_name=cfg["model_name"],
                    messages=messages,
                    max_tokens=cfg["max_tokens"],
                    temperature=cfg["temperature"],
                    top_p=cfg["top_p"],
                )
                pred_idx = parse_choice(text, len(opts))

    except Exception as e:
        error = str(e)

    correct = (pred_idx is not None and gt_idx == pred_idx)

    return {
        "model": cfg["model_name"],
        "file": cfg["infile"],
        "token": token,
        "type": typ,
        "category": cat,
        "waymo_split": cfg["split_name"],
        "n_options": len(opts),
        "gt_index": gt_idx,
        "pred_index": pred_idx,
        "correct": bool(correct),
        "paths": used_path,
        "error": error,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    # vLLM OpenAI-compatible endpoint
    ap.add_argument("--api-base", default="http://localhost:8001/v1",
                    help="OpenAI-compatible server base URL (vLLM default: http://localhost:8001/v1).")
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"),
                    help="API key string for OpenAI SDK. vLLM usually accepts any string.")

    ap.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--jsonl", default=DEFAULT_JSONL)

    ap.add_argument("--img-root", default=DEFAULT_IMG_ROOT,
                    help="Directory containing ImageQA images: token_XXX_CAM.jpg (6 views).")
    ap.add_argument("--mosaic-root", default=DEFAULT_MOSAIC_ROOT,
                    help="Directory containing VideoQA mosaic frames: token_XXX.jpg")

    ap.add_argument("--split-name", default="validation",
                    help="Value to write into waymo_split column (e.g., train/validation/test).")

    ap.add_argument("--answer-index-base", type=int, default=0, choices=[0, 1])

    ap.add_argument("--video-stride", type=int, default=5,
                    help="Sample every N-th mosaic frame for VideoQA (adjust based on your GPU memory).")
    ap.add_argument("--video-max-frames", type=int, default=60,
                    help="Max mosaic frames per VideoQA sample (-1 means unlimited; can be huge).")

    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--save-dir", default="runs_vllm")

    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--flush-every", type=int, default=200)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=5)

    args = ap.parse_args()

    img_root = Path(args.img_root)
    mosaic_root = Path(args.mosaic_root)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    infile = Path(args.jsonl)
    items: List[Dict] = []
    if infile.exists():
        with open(infile, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    else:
        print(f"[WARN] Not found: {infile}")
        return

    if args.max_examples > 0:
        items = items[:args.max_examples]

    total_lines = len(items)
    print(f"[INFO] Total items: {total_lines}")

    cfg = {
        "api_base": args.api_base,
        "api_key": args.api_key,
        "model_name": args.model_name,
        "img_root": img_root,
        "mosaic_root": mosaic_root,
        "answer_index_base": int(args.answer_index_base),
        "video_stride": int(args.video_stride),
        "video_max_frames": int(args.video_max_frames),
        "infile": str(infile),
        "split_name": str(args.split_name),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_tokens),
    }

    csv_path = out_dir / f"pred_{args.model_name.replace('/','_')}_{args.split_name}.csv"
    header = ["model", "file", "token", "type", "category", "waymo_split", "n_options",
              "gt_index", "pred_index", "correct", "paths", "error"]
    csv_lock = threading.Lock()

    if total_lines == 0:
        print("[WARN] No items to process")
        return

    written = 0
    with tqdm(total=total_lines, desc=f"Evaluating {args.model_name} ({args.split_name})", unit="sample") as pbar:
        with ThreadPoolExecutor(max_workers=int(args.num_workers)) as ex:
            futures = [ex.submit(process_one, it, cfg) for it in items]

            buffer_rows: List[Dict] = []
            for fut in as_completed(futures):
                rec = fut.result()
                if not rec.get("skip"):
                    buffer_rows.append(rec)
                    written += 1
                    if len(buffer_rows) >= int(args.flush_every):
                        for r in buffer_rows:
                            csv_append_row(csv_path, header, r, csv_lock)
                        buffer_rows.clear()
                pbar.update(1)

            if buffer_rows:
                for r in buffer_rows:
                    csv_append_row(csv_path, header, r, csv_lock)

    print(f"[SAVE] {csv_path} (appended {written} rows)")

    # Summary
    df = pd.read_csv(csv_path)
    df = df[df["model"] == args.model_name].copy()

    def _acc(g: pd.DataFrame) -> float:
        sub = g[g["pred_index"].notna()]
        if len(sub) == 0:
            return 0.0
        return float(sub["correct"].sum()) / float(len(sub))

    if len(df) == 0:
        overall_acc = 0.0
        by_type = pd.DataFrame({"type": [], "acc": []})
        img_cat = pd.DataFrame({"category": [], "acc": []})
        vid_cat = pd.DataFrame({"category": [], "acc": []})
    else:
        overall_acc = _acc(df)
        by_type = df.groupby("type", dropna=False).apply(_acc).rename("acc").reset_index()
        img_cat = df[df["type"] == "image"].groupby("category", dropna=False).apply(_acc).rename("acc").reset_index()
        vid_cat = df[df["type"] == "video"].groupby("category", dropna=False).apply(_acc).rename("acc").reset_index()

    print(f"\n=== [{args.split_name.upper()}] Overall ACC (pred not None) === {overall_acc:.4f}")
    print("\n=== Accuracy by type ===")
    print(by_type)
    print("\n=== Accuracy by IMAGE category ===")
    print(img_cat.sort_values("acc", ascending=False))
    print("\n=== Accuracy by VIDEO category ===")
    print(vid_cat.sort_values("acc", ascending=False))

    summ_path = out_dir / f"summary_{args.model_name.replace('/','_')}_{args.split_name}.txt"
    with open(summ_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Split: {args.split_name}\n")
        f.write(f"Overall ACC: {overall_acc:.4f}\n\n")
        f.write("[By type]\n")
        f.write(by_type.to_string(index=False))
        f.write("\n\n[By IMAGE category]\n")
        f.write(img_cat.sort_values("acc", ascending=False).to_string(index=False))
        f.write("\n\n[By VIDEO category]\n")
        f.write(vid_cat.sort_values("acc", ascending=False).to_string(index=False))

    print(f"[SAVE] {summ_path}\nDone.")


if __name__ == "__main__":
    main()
