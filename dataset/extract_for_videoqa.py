#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import Dict

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2

# Less TF spam
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Export 3x3 mosaics for VIDEO-QA tokens from Waymo E2E TFRecords (RAW tiles, NO resize/crop, with labels)."
    )

    p.add_argument("--dataset-root", type=str,
                   default="/raid/workspace/sjyu/waymo/raw_data",
                   help="Waymo dataset root (contains train.tfrecord/validation.tfrecord/test.tfrecord or sharded dirs).")
    p.add_argument("--split", type=str, default="train",
                   choices=["train", "validation", "test"], help="Dataset split.")
    p.add_argument("--tfrecord-path", type=str, default=None,
                   help=("Path to TFRecord file or directory. "
                         "If omitted, defaults to <dataset-root>/<split>.tfrecord"))

    p.add_argument("--target-jsonl", type=str, required=True,
                   help="JSONL file containing samples. We will collect tokens with type=video.")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to save mosaic images.")

    # TF runtime knobs (optional)
    p.add_argument("--tf-mem-growth", action="store_true",
                   help="Enable TF memory growth on the first visible GPU.")
    p.set_defaults(tf_mem_growth=True)

    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality.")

    # performance / safety
    p.add_argument("--skip-count", action="store_true",
                   help="Skip counting TFRecord frames first.")
    p.add_argument("--max-video-tokens", type=int, default=-1,
                   help="Limit number of unique video tokens loaded from JSONL (-1 means all).")
    p.add_argument("--max-frames-per-token", type=int, default=-1,
                   help="Limit number of frames exported per token (-1 means all).")
    p.add_argument("--stride", type=int, default=1,
                   help="Export every N-th frame for video tokens (default: 1).")
    p.add_argument("--pad-len", type=int, default=3,
                   help="Zero-padding length for frame index in filename (default: 3 -> 039).")

    # label style
    p.add_argument("--label-font-scale", type=float, default=0.7,
                   help="Font scale for camera labels (default: 0.7).")
    p.add_argument("--label-thickness", type=int, default=2,
                   help="Line thickness for camera labels (default: 2).")
    p.add_argument("--label-pad", type=int, default=6,
                   help="Padding for label background box (default: 6).")

    # layout
    p.add_argument("--gap", type=int, default=8,
                   help="Gap (pixels) between tiles in the mosaic (default: 8).")
    p.add_argument("--tile-align", choices=["topleft", "center"], default="center",
                   help="How to align each raw tile inside its cell (default: center).")

    # split filter in jsonl
    p.add_argument("--allow-splits", type=str, nargs="*",
                   default=["train", "validation", "val", "valid", "dev", "test"],
                   help="Allowed waymo_split values inside JSONL.")

    return p.parse_args()


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
CAM_ID_TO_NAME = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
    6: "REAR_LEFT",
    7: "REAR",
    8: "REAR_RIGHT",
}

MOSAIC_GRID = [
    [2, 1, 3],
    [4, 0, 5],  # 0 means blank
    [6, 7, 8],
]

EXPECTED_CAM_IDS = [1, 2, 3, 4, 5, 6, 7, 8]


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
def setup_tf(mem_growth: bool):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus and mem_growth:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except Exception:
            pass


def resolve_tfrecord_shards(dataset_root: str, split: str, tfrecord_path: str | None):
    if tfrecord_path is None:
        tfrecord_path = os.path.join(dataset_root, f"{split}.tfrecord")

    if os.path.isdir(tfrecord_path):
        shards = tf.io.gfile.glob(os.path.join(tfrecord_path, "*.tfrecord-*"))
    elif os.path.isfile(tfrecord_path):
        shards = [tfrecord_path]
    else:
        raise FileNotFoundError(f"TFRecord path does not exist: {tfrecord_path}")

    if not shards:
        raise FileNotFoundError(f"No TFRecord shards found from: {tfrecord_path}")

    return shards, tfrecord_path


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def padn(n: int, pad_len: int) -> str:
    return str(int(n)).zfill(int(pad_len))


def parse_token_and_frame_idx(context_name: str):
    if not context_name:
        return None, None

    m = re.match(r"^([^-]+)-(\d+)", context_name)
    if m:
        token = m.group(1)
        fidx = int(m.group(2).lstrip("0") or "0")
        return token, fidx

    parts = context_name.split("-")
    token = parts[0]
    for p in parts[1:]:
        if p.isdigit():
            return token, int(p.lstrip("0") or "0")
    return token, None


def load_video_tokens_from_jsonl(jsonl_path, allow_splits, max_video_tokens=-1):
    allow_splits = set(allow_splits) | {None}
    video_tokens = []
    seen = set()
    line_cnt = 0
    video_rows = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line_cnt += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if str(obj.get("type", "")).lower() != "video":
                continue

            split = obj.get("waymo_split")
            if split not in allow_splits:
                continue

            token = obj.get("token")
            if not token:
                continue

            video_rows += 1
            if token not in seen:
                seen.add(token)
                video_tokens.append(token)
                if max_video_tokens > 0 and len(video_tokens) >= max_video_tokens:
                    break

    return set(video_tokens), {"lines": line_cnt, "video_rows": video_rows, "video_tokens": len(video_tokens)}


def cam_label(cid: int) -> str:
    return CAM_ID_TO_NAME.get(cid, f"CAM_{cid}").replace("_", " ")


def draw_label(tile_rgb: np.ndarray, text: str, font_scale: float, thickness: int, pad: int) -> np.ndarray:
    img_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x0, y0 = pad, pad
    x1, y1 = x0 + tw + pad * 2, y0 + th + baseline + pad * 2

    cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    tx, ty = x0 + pad, y0 + pad + th
    cv2.putText(img_bgr, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def place_on_canvas(canvas: np.ndarray, tile: np.ndarray, x: int, y: int, align: str):
    """
    Place tile (RGB) onto canvas (RGB) at cell top-left (x,y), with optional alignment inside the cell.
    Canvas area for the cell is inferred by canvas bounds; we just place tile at computed offset.
    """
    ch, cw = canvas.shape[:2]
    th, tw = tile.shape[:2]

    # x,y are the top-left of the cell
    # The cell size is not passed here; we will place relative to given x,y with offsets computed outside.
    canvas[y:y+th, x:x+tw] = tile


def build_raw_mosaic_with_padding(
    tiles_rgb: Dict[int, np.ndarray],
    font_scale: float,
    thickness: int,
    pad: int,
    gap: int,
    align: str,
):
    """
    Build 3x3 mosaic WITHOUT resizing/cropping any tile.
    - Each cell size = max(H), max(W) among tiles assigned to that row/col (excluding blank).
    - Tiles are placed with padding (black) inside each cell.
    - Gaps between cells are inserted.
    """
    blank_color = (0, 0, 0)

    # Apply labels first (still no resize/crop)
    labeled = {}
    for cid, img in tiles_rgb.items():
        labeled[cid] = draw_label(img, cam_label(cid), font_scale, thickness, pad)

    # Determine cell widths per column and heights per row
    row_heights = [0, 0, 0]
    col_widths = [0, 0, 0]

    for r in range(3):
        for c in range(3):
            cid = MOSAIC_GRID[r][c]
            if cid == 0:
                continue
            im = labeled[cid]
            h, w = im.shape[:2]
            row_heights[r] = max(row_heights[r], h)
            col_widths[c] = max(col_widths[c], w)

    # If some row/col is all blank (unlikely), set to 1 to avoid zero-sized canvas
    row_heights = [h if h > 0 else 1 for h in row_heights]
    col_widths = [w if w > 0 else 1 for w in col_widths]

    H = sum(row_heights) + gap * 2
    W = sum(col_widths) + gap * 2

    mosaic = np.zeros((H, W, 3), dtype=np.uint8)
    mosaic[:, :] = blank_color

    # Place each tile into its cell with padding
    y = 0
    for r in range(3):
        x = 0
        cell_h = row_heights[r]
        for c in range(3):
            cell_w = col_widths[c]
            cid = MOSAIC_GRID[r][c]

            if cid != 0:
                im = labeled[cid]
                h, w = im.shape[:2]

                if align == "center":
                    oy = y + (cell_h - h) // 2
                    ox = x + (cell_w - w) // 2
                else:  # topleft
                    oy, ox = y, x

                mosaic[oy:oy+h, ox:ox+w] = im

            x += cell_w
            if c < 2:
                x += gap
        y += cell_h
        if r < 2:
            y += gap

    return mosaic


def save_jpg(out_path, img_rgb, jpeg_quality=95):
    return cv2.imwrite(
        out_path,
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()
    setup_tf(mem_growth=args.tf_mem_growth)

    os.makedirs(args.output_dir, exist_ok=True)

    shards, resolved_tfrecord_path = resolve_tfrecord_shards(
        dataset_root=args.dataset_root,
        split=args.split,
        tfrecord_path=args.tfrecord_path,
    )

    video_tokens, stats = load_video_tokens_from_jsonl(
        jsonl_path=args.target_jsonl,
        allow_splits=args.allow_splits,
        max_video_tokens=args.max_video_tokens,
    )

    if not video_tokens:
        raise SystemExit("[WARN] No video tokens found in JSONL (type=video).")

    print(f"[INFO] TFRecord source: {resolved_tfrecord_path}")
    print(f"[INFO] JSONL lines: {stats['lines']}, video rows: {stats['video_rows']}, unique video tokens: {stats['video_tokens']}")
    print(f"[INFO] Output dir: {args.output_dir}")
    print(f"[INFO] Layout: FL/F/FR | SL/_/SR | RL/R/RR")
    print(f"[INFO] RAW tiles: no resize/crop, padding allowed")
    print(f"[INFO] Labels: enabled")

    total_frames = None
    if not args.skip_count:
        count_ds = tf.data.TFRecordDataset(shards, compression_type="")
        total_frames = sum(1 for _ in count_ds)
        print(f"[INFO] Total TFRecord frames: {total_frames}")

    exported_per_token = {t: 0 for t in video_tokens}

    ds = tf.data.TFRecordDataset(shards, compression_type="")
    pbar = tqdm(ds, total=total_frames, desc=f"[{args.split}] exporting mosaics", unit="frame")

    saved = 0
    skipped_missing_cam = 0

    for raw in pbar:
        frame_msg = wod_e2ed_pb2.E2EDFrame()
        frame_msg.ParseFromString(raw.numpy())

        token, fidx = parse_token_and_frame_idx(frame_msg.frame.context.name)
        if token is None or fidx is None:
            continue

        if token not in video_tokens:
            continue

        if args.stride > 1 and (fidx % args.stride != 0):
            continue

        if args.max_frames_per_token > 0 and exported_per_token[token] >= args.max_frames_per_token:
            continue

        frame_cam_map = {cam.name: cam for cam in frame_msg.frame.images}
        missing = [cid for cid in EXPECTED_CAM_IDS if cid not in frame_cam_map]
        if missing:
            skipped_missing_cam += 1
            pbar.set_postfix(saved=saved, miss_cam=skipped_missing_cam)
            continue

        # decode 8 cams (RAW)
        tiles_rgb = {}
        ok = True
        for cid in EXPECTED_CAM_IDS:
            cam = frame_cam_map[cid]
            im_bgr = cv2.imdecode(np.frombuffer(cam.image, np.uint8), cv2.IMREAD_COLOR)
            if im_bgr is None:
                ok = False
                break
            tiles_rgb[cid] = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        if not ok or len(tiles_rgb) != 8:
            skipped_missing_cam += 1
            pbar.set_postfix(saved=saved, miss_cam=skipped_missing_cam)
            continue

        mosaic_rgb = build_raw_mosaic_with_padding(
            tiles_rgb=tiles_rgb,
            font_scale=args.label_font_scale,
            thickness=args.label_thickness,
            pad=args.label_pad,
            gap=args.gap,
            align=args.tile_align,
        )

        fidx_str = padn(fidx, args.pad_len)
        out_name = f"{token}_{fidx_str}.jpg"
        out_path = os.path.join(args.output_dir, out_name)

        if save_jpg(out_path, mosaic_rgb, jpeg_quality=args.jpeg_quality):
            saved += 1
            exported_per_token[token] += 1

        pbar.set_postfix(saved=saved, miss_cam=skipped_missing_cam)

    pbar.close()
    print(f"✅ Done: saved {saved} mosaics")
    print(f"[INFO] skipped_missing_cam_frames: {skipped_missing_cam}")


if __name__ == "__main__":
    main()
