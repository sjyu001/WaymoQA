#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, Iterator, List, Tuple, Any, Optional


DEFAULT_CAM_ORDER = [
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
    "SIDE_RIGHT",
    "SIDE_LEFT",
    "REAR_LEFT",
    "REAR_RIGHT",
    "REAR",
]


# ------------------------------------------------------------
# IO
# ------------------------------------------------------------
def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def zero_pad(n: Any, pad_len: int) -> Optional[str]:
    if n is None:
        return None
    try:
        return f"{int(n):0{pad_len}d}"
    except Exception:
        return str(n)


def make_image_names(token: str, frame_str: str, cams: List[str], ext: str) -> List[str]:
    # ImageQA: token_frame_CAM.jpg
    return [f"{token}_{frame_str}_{cam}.{ext}" for cam in cams]


def make_video_mosaic_name(token: str, frame_str: str, ext: str) -> str:
    # VideoQA mosaics: token_frame.jpg
    return f"{token}_{frame_str}.{ext}"


# ------------------------------------------------------------
# QA normalization (train vs validation)
# ------------------------------------------------------------
def normalize_qa(ex: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Supports:
      - train: {question, answer}
      - validation: {question, options, answer_index}
    Produces:
      question_text (with options appended if present),
      answer_text
    """
    q = str(ex.get("question", "")).strip()
    if not q:
        return None, None

    options = ex.get("options")
    if isinstance(options, list) and len(options) > 0:
        opt_lines = "\n".join(str(o).strip() for o in options if str(o).strip())
        if opt_lines:
            q = f"{q}\n\nOptions:\n{opt_lines}"

    a = ex.get("answer")
    if a is not None and str(a).strip():
        return q, str(a).strip()

    ans_idx = ex.get("answer_index")
    if isinstance(options, list) and ans_idx is not None:
        try:
            ans_idx = int(ans_idx)
            if 0 <= ans_idx < len(options):
                return q, str(options[ans_idx]).strip()
        except Exception:
            pass

    return None, None


# ------------------------------------------------------------
# Video mosaics: list frames from disk, then stride-sample
# ------------------------------------------------------------
def parse_frame_idx_from_name(token: str, filename: str) -> Optional[int]:
    # expects: {token}_{digits}.jpg (or other ext)
    m = re.match(rf"^{re.escape(token)}_(\d+)\.", filename)
    if not m:
        return None
    return int(m.group(1))


def list_video_frames_from_disk(token: str, mosaic_dir: Path, ext: str) -> List[int]:
    paths = sorted(mosaic_dir.glob(f"{token}_*.{ext}"))
    idxs = []
    for p in paths:
        fi = parse_frame_idx_from_name(token, p.name)
        if fi is not None:
            idxs.append(fi)
    return sorted(set(idxs))


def sample_by_stride(frames: List[int], stride: int, max_frames: int) -> List[int]:
    if not frames:
        return []
    stride = max(1, int(stride))
    sampled = frames[0::stride]
    if max_frames > 0:
        sampled = sampled[:max_frames]
    return sampled


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Build LLaVA-style conversation JSON (filenames only). "
                    "ImageQA uses 8-view images; VideoQA uses mosaic frames sampled from disk."
    )
    ap.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files.")
    ap.add_argument("--out", required=True, help="Output JSON path.")

    # ImageQA naming
    ap.add_argument("--image-ext", default="jpg", help="Extension for ImageQA camera images (default: jpg).")
    ap.add_argument("--pad-len", type=int, default=3, help="Zero-padding length (default: 3 -> 039).")
    ap.add_argument("--cam-order", nargs="+", default=DEFAULT_CAM_ORDER,
                    help="Camera name order for ImageQA (default: 8-cam).")

    # VideoQA mosaic lookup (disk)
    ap.add_argument("--video-mosaic-dir", required=True,
                    help="Directory containing mosaic frames named {token}_{idx}.jpg")
    ap.add_argument("--video-mosaic-ext", default="jpg",
                    help="Extension for mosaic frames (default: jpg).")
    ap.add_argument("--video-stride", type=int, default=5,
                    help="Sample every N-th available mosaic frame (default: 5).")
    ap.add_argument("--video-max-frames", type=int, default=60,
                    help="Max mosaics per token (default: 60). Set -1 for unlimited.")
    ap.add_argument("--video-min-frames", type=int, default=1,
                    help="Skip token if fewer mosaics than this (default: 1).")

    ap.add_argument("--only-type", choices=["image", "video", "all"], default="all",
                    help="Process only image, only video, or all (default: all).")
    ap.add_argument("--repeat-media-tokens", action="store_true",
                    help="Repeat <image> tokens for every question turn (default: only first turn).")

    return ap.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()

    # (token, frame_str) -> [(q,a)]
    img_groups: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    # token -> [(q,a)]
    vid_groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    total_read = 0
    total_used = 0

    for p in args.inputs:
        for ex in iter_jsonl(p):
            total_read += 1

            typ = str(ex.get("type", "")).lower()
            token = ex.get("token")
            if not token:
                continue

            if args.only_type != "all" and typ != args.only_type:
                continue

            q, a = normalize_qa(ex)
            if not q or not a:
                continue

            if typ == "image":
                frame_idx = ex.get("frame_index")
                frame_str = zero_pad(frame_idx, args.pad_len)
                if frame_str is None:
                    continue
                img_groups[(token, frame_str)].append((q, a))
                total_used += 1

            elif typ == "video":
                vid_groups[token].append((q, a))
                total_used += 1

    out_items: List[OrderedDict] = []

    # -------------------------
    # IMAGE items
    # -------------------------
    if args.only_type in ("all", "image"):
        for (token, frame_str), qa_list in img_groups.items():
            images = make_image_names(token, frame_str, args.cam_order, args.image_ext)

            conv = []
            image_tokens = "\n".join(["<image>"] * len(images))

            first = True
            for q, a in qa_list:
                if args.repeat_media_tokens or first:
                    conv.append({"from": "human", "value": f"{image_tokens}\n\n{q}"})
                    first = False
                else:
                    conv.append({"from": "human", "value": q})
                conv.append({"from": "gpt", "value": a})

            out_items.append(OrderedDict([
                ("id", f"{token}_{frame_str}"),
                ("image", images),
                ("conversations", conv),
            ]))

    # -------------------------
    # VIDEO items -> mosaic frames from disk (filenames only)
    # -------------------------
    if args.only_type in ("all", "video"):
        mosaic_dir = Path(args.video_mosaic_dir)

        for token, qa_list in vid_groups.items():
            available = list_video_frames_from_disk(token, mosaic_dir, args.video_mosaic_ext)
            sampled = sample_by_stride(available, args.video_stride, args.video_max_frames)

            if len(sampled) < args.video_min_frames:
                continue

            mosaics = []
            for fi in sampled:
                frame_str = zero_pad(fi, args.pad_len)
                if frame_str is None:
                    continue
                mosaics.append(make_video_mosaic_name(token, frame_str, args.video_mosaic_ext))

            conv = []
            image_tokens = "\n".join(["<image>"] * len(mosaics)) if mosaics else ""

            first = True
            for q, a in qa_list:
                if args.repeat_media_tokens or first:
                    if image_tokens:
                        conv.append({"from": "human", "value": f"{image_tokens}\n\n{q}"})
                    else:
                        conv.append({"from": "human", "value": q})
                    first = False
                else:
                    conv.append({"from": "human", "value": q})
                conv.append({"from": "gpt", "value": a})

            out_items.append(OrderedDict([
                ("id", token),
                ("image", mosaics),  # VideoQA represented as a list of mosaic JPG filenames
                ("conversations", conv),
            ]))

    out_items.sort(key=lambda x: x["id"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    print(f"[OK] read={total_read}, used={total_used}, wrote={len(out_items)} -> {out_path}")


if __name__ == "__main__":
    main()
