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
        description="(GPU-default) Export 8 camera images from Waymo E2E TFRecords using JSONL targets."
    )

    p.add_argument("--dataset-root", type=str,
                   default="./dataset/waymoe2e",
                   help="Waymo dataset root (contains train.tfrecord/validation.tfrecord/test.tfrecord or sharded dirs).")
    p.add_argument("--split", type=str, default="train",
                   choices=["train", "validation", "test"], help="Dataset split.")
    p.add_argument("--tfrecord-path", type=str, default=None,
                   help=("Path to TFRecord file or directory. "
                         "If omitted, defaults to <dataset-root>/<split>.tfrecord"))

    p.add_argument("--target-jsonl", type=str, required=True,
                   help="JSONL file containing token/frame_index targets.")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to save exported images.")

    # TF runtime knobs (optional)
    p.add_argument("--tf-mem-growth", action="store_true",
                   help="Enable TF memory growth on the first visible GPU.")
    p.set_defaults(tf_mem_growth=True)

    p.add_argument("--undistort", action="store_true",
                   help="Apply undistortion using camera intrinsics.")
    p.add_argument("--no-undistort", dest="undistort", action="store_false",
                   help="Disable undistortion.")
    p.set_defaults(undistort=True)

    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality.")
    p.add_argument("--max-targets", type=int, default=-1,
                   help="Process only the first N targets from JSONL (-1 means all).")
    p.add_argument("--skip-count", action="store_true",
                   help="Skip counting TFRecord frames first.")
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
def pad3(n: int) -> str:
    return str(n).zfill(3)


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


def load_targets_from_jsonl(jsonl_path, allow_splits, max_targets=-1):
    allow_splits = set(allow_splits) | {None}
    targets = set()
    line_cnt = 0

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

            if obj.get("type") != "image":
                continue

            split = obj.get("waymo_split")
            if split not in allow_splits:
                continue

            token = obj.get("token")
            fidx = obj.get("frame_index")
            if token is None or fidx is None:
                continue

            targets.add((token, int(fidx)))

            if max_targets > 0 and len(targets) >= max_targets:
                break

    return targets, line_cnt


def save_image(out_path, img_bgr, jpeg_quality=95):
    return cv2.imwrite(out_path, img_bgr,
                       [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])


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

    targets, line_cnt = load_targets_from_jsonl(
        jsonl_path=args.target_jsonl,
        allow_splits=args.allow_splits,
        max_targets=args.max_targets,
    )

    if not targets:
        raise SystemExit(
            "[WARN] No valid (token, frame_index) targets found in JSONL. "
            "Check waymo_split / token / frame_index / type fields."
        )

    print(f"[INFO] TFRecord source: {resolved_tfrecord_path}")
    print(f"[INFO] JSONL lines: {line_cnt}")
    print(f"[INFO] Unique targets: {len(targets)}")
    print(f"[INFO] Output dir: {args.output_dir}")
    print(f"[INFO] Undistort: {args.undistort}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    total_frames = None
    if not args.skip_count:
        count_ds = tf.data.TFRecordDataset(shards, compression_type="")
        total_frames = sum(1 for _ in count_ds)
        print(f"[INFO] Total TFRecord frames: {total_frames}")

    remaining = set(targets)
    saved_count = 0
    found_frames = 0
    skipped_missing_cam = 0

    ds = tf.data.TFRecordDataset(shards, compression_type="")
    pbar = tqdm(ds, total=total_frames, desc=f"[{args.split}] scanning TFRecords", unit="frame")

    for raw in pbar:
        if not remaining:
            break

        frame_msg = wod_e2ed_pb2.E2EDFrame()
        frame_msg.ParseFromString(raw.numpy())

        token, fidx = parse_token_and_frame_idx(frame_msg.frame.context.name)
        if token is None or fidx is None:
            continue

        key = (token, fidx)
        if key not in remaining:
            continue

        found_frames += 1
        cal_map = build_calib_map(frame_msg)
        frame_cam_map = {cam.name: cam for cam in frame_msg.frame.images}

        missing = [cid for cid in EXPECTED_CAM_IDS if cid not in frame_cam_map]
        if missing:
            skipped_missing_cam += 1
            remaining.remove(key)
            pbar.set_postfix(remaining=len(remaining), saved=saved_count, found=found_frames,
                             missing_cam_frames=skipped_missing_cam)
            continue

        saved_any = False
        fidx_str = pad3(fidx)

        for cam_id in EXPECTED_CAM_IDS:
            cam = frame_cam_map[cam_id]
            cam_name = CAM_ID_TO_NAME.get(cam_id, f"CAM_{cam_id}")

            img_bgr = cv2.imdecode(np.frombuffer(cam.image, np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            if args.undistort:
                cal = cal_map.get(cam_id)
                if cal is not None:
                    try:
                        img_bgr = undistort_with_cal(img_bgr, cal)
                    except Exception:
                        pass

            out_name = f"{token}_{fidx_str}_{cam_name}.jpg"
            out_path = os.path.join(args.output_dir, out_name)

            if save_image(out_path, img_bgr, jpeg_quality=args.jpeg_quality):
                saved_count += 1
                saved_any = True

        if saved_any:
            remaining.remove(key)

        pbar.set_postfix(remaining=len(remaining), saved=saved_count, found=found_frames,
                         missing_cam_frames=skipped_missing_cam)

    pbar.close()

    print(f"✅ Done: saved {saved_count} images")
    print(f"[INFO] Found target frames: {found_frames}")
    print(f"[INFO] Remaining unmatched targets: {len(remaining)}")
    print(f"[INFO] Frames skipped due to missing cameras: {skipped_missing_cam}")


if __name__ == "__main__":
    main()
