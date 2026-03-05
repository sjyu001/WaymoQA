#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse

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
        description="Export 3x3 mosaics for VIDEO-QA tokens from Waymo E2E TFRecords."
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

    # image processing
    p.add_argument("--undistort", action="store_true",
                   help="Apply undistortion using camera intrinsics.")
    p.add_argument("--no-undistort", dest="undistort", action="store_false",
                   help="Disable undistortion.")
    p.set_defaults(undistort=True)

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

    # split filter in jsonl
    p.add_argument("--allow-splits", type=str, nargs="*",
                   default=["train", "validation", "val", "valid", "dev", "test"],
                   help="Allowed waymo_split values inside JSONL.")

    return p.parse_args()


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
# Waymo CameraName enum (commonly)
# 1: FRONT, 2: FRONT_LEFT, 3: FRONT_RIGHT,
# 4: SIDE_LEFT, 5: SIDE_RIGHT,
# 6: REAR_LEFT, 7: REAR, 8: REAR_RIGHT
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

# Mosaic layout (3x3 with center blank):
# [ FRONT_LEFT, FRONT, FRONT_RIGHT ]
# [ SIDE_LEFT , blank, SIDE_RIGHT  ]
# [ REAR_LEFT , REAR , REAR_RIGHT  ]
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


def build_calib_map(frame_msg):
    return {cal.name: cal for cal in frame_msg.frame.context.camera_calibrations}


def undistort_with_cal(img_bgr, cal):
    intr = np.asarray(cal.intrinsic, dtype=np.float32)
    if intr.size < 9:
        return img_bgr

    fu, fv, cu, cv = intr[:4]
    k1, k2, p1, p2, k3 = intr[4:9]

    K = np.array([[fu, 0, cu],
                  [0, fv, cv],
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    h, w = img_bgr.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
    return cv2.undistort(img_bgr, K, D, None, new_K)


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


def normalize_tiles_to_same_size(imgs_bgr: dict):
    """
    imgs_bgr: {cam_id: image_bgr}
    -> returns {cam_id: tile_rgb} where every tile has the same (H, W, 3)
    Strategy:
      1) pick target_h = min heights across available cams
      2) resize each to target_h keeping aspect ratio
      3) crop all to target_w = min widths after resize
      4) convert to RGB
    """
    heights = [im.shape[0] for im in imgs_bgr.values()]
    target_h = min(heights)

    resized = {}
    widths = []
    for cid, im in imgs_bgr.items():
        h, w = im.shape[:2]
        new_w = max(1, int(w * (target_h / h)))
        im_rs = cv2.resize(im, (new_w, target_h))
        resized[cid] = im_rs
        widths.append(new_w)

    target_w = min(widths)
    tiles_rgb = {}
    for cid, im_rs in resized.items():
        im_crop = im_rs[:, :target_w]
        tiles_rgb[cid] = cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB)

    return tiles_rgb, target_h, target_w


def build_3x3_mosaic(tiles_rgb: dict, tile_h: int, tile_w: int):
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    rows = []
    for row in MOSAIC_GRID:
        row_tiles = []
        for cid in row:
            if cid == 0:
                row_tiles.append(blank)
            else:
                row_tiles.append(tiles_rgb[cid])
        rows.append(np.concatenate(row_tiles, axis=1))

    mosaic = np.concatenate(rows, axis=0)  # RGB
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
    print(f"[INFO] Undistort: {args.undistort}")
    print(f"[INFO] Layout: FL/F/FR | SL/_/SR | RL/R/RR")

    total_frames = None
    if not args.skip_count:
        count_ds = tf.data.TFRecordDataset(shards, compression_type="")
        total_frames = sum(1 for _ in count_ds)
        print(f"[INFO] Total TFRecord frames: {total_frames}")

    # per token frame limits
    exported_per_token = {t: 0 for t in video_tokens}

    ds = tf.data.TFRecordDataset(shards, compression_type="")
    pbar = tqdm(ds, total=total_frames, desc=f"[{args.split}] exporting mosaics", unit="frame")

    saved = 0
    skipped_missing_cam = 0
    scanned = 0

    for raw in pbar:
        scanned += 1
        frame_msg = wod_e2ed_pb2.E2EDFrame()
        frame_msg.ParseFromString(raw.numpy())

        token, fidx = parse_token_and_frame_idx(frame_msg.frame.context.name)
        if token is None or fidx is None:
            continue

        if token not in video_tokens:
            continue

        # stride
        if args.stride > 1 and (fidx % args.stride != 0):
            continue

        # cap per token
        if args.max_frames_per_token > 0 and exported_per_token[token] >= args.max_frames_per_token:
            continue

        frame_cam_map = {cam.name: cam for cam in frame_msg.frame.images}
        missing = [cid for cid in EXPECTED_CAM_IDS if cid not in frame_cam_map]
        if missing:
            skipped_missing_cam += 1
            pbar.set_postfix(saved=saved, miss_cam=skipped_missing_cam)
            continue

        cal_map = build_calib_map(frame_msg)

        # decode 8 cams
        imgs_bgr = {}
        for cid in EXPECTED_CAM_IDS:
            cam = frame_cam_map[cid]
            im = cv2.imdecode(np.frombuffer(cam.image, np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                imgs_bgr = {}
                break
            if args.undistort:
                cal = cal_map.get(cid)
                if cal is not None:
                    try:
                        im = undistort_with_cal(im, cal)
                    except Exception:
                        pass
            imgs_bgr[cid] = im

        if len(imgs_bgr) != 8:
            skipped_missing_cam += 1
            pbar.set_postfix(saved=saved, miss_cam=skipped_missing_cam)
            continue

        tiles_rgb, tile_h, tile_w = normalize_tiles_to_same_size(imgs_bgr)
        mosaic_rgb = build_3x3_mosaic(tiles_rgb, tile_h, tile_w)

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
