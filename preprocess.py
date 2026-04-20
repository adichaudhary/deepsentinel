"""
preprocess.py — DeepSentinel face crop extraction pipeline

Usage:
    python preprocess.py \
        --real_dir   data/raw/real  \
        --fake_dir   data/raw/fake  \
        --out_real   data/crops/real \
        --out_fake   data/crops/fake \
        --frame_step 10 \
        --conf_thresh 0.95 \
        --padding 0.20

Filename convention:  {label}__{video_stem}__{frame_idx:06d}__{face_idx}.jpg
  e.g.  real__aabqzsgz__000120__0.jpg
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def make_mtcnn(device: str = "cpu") -> MTCNN:
    return MTCNN(
        keep_all=True,
        device=device,
        select_largest=False,
        post_process=False,
        min_face_size=40,
    )


def padded_crop(
    frame_rgb: np.ndarray,
    box: list[float],
    padding: float = 0.20,
) -> np.ndarray | None:
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    pad_x = bw * padding
    pad_y = bh * padding
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))

    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def extract_faces_from_video(
    video_path: Path,
    out_dir: Path,
    label: str,
    frame_step: int = 10,
    conf_thresh: float = 0.95,
    padding: float = 0.20,
    device: str = "cpu",
) -> int:
    mtcnn = make_mtcnn(device)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("Cannot open %s — skipping", video_path.name)
        return 0

    video_stem = video_path.stem
    saved = 0
    frame_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            boxes, probs = mtcnn.detect(pil)

            if boxes is not None:
                for face_idx, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob is None or prob < conf_thresh:
                        continue
                    crop = padded_crop(rgb, box, padding)
                    if crop is None:
                        continue

                    fname = f"{label}__{video_stem}__{frame_idx:06d}__{face_idx}.jpg"
                    out_path = out_dir / fname
                    Image.fromarray(crop).save(out_path, quality=95)
                    saved += 1

        frame_idx += 1

    cap.release()
    return saved


# ---------------------------------------------------------------------------
# Per-label task collection + parallel dispatch
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> tuple[str, str, int]:
    video_path, out_dir, label, frame_step, conf_thresh, padding, device = args
    count = extract_faces_from_video(
        video_path, out_dir, label, frame_step, conf_thresh, padding, device
    )
    return video_path.name, label, count


def collect_tasks(
    in_dir: Path,
    out_dir: Path,
    label: str,
    frame_step: int,
    conf_thresh: float,
    padding: float,
    device: str,
) -> list[tuple]:
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = [p for p in in_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        log.warning("No videos found in %s", in_dir)
    return [
        (v, out_dir, label, frame_step, conf_thresh, padding, device)
        for v in videos
    ]


def run_parallel(
    all_tasks: list[tuple],
    workers: int,
) -> tuple[int, int]:
    real_saved = fake_saved = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, t): t for t in all_tasks}

        with tqdm(
            total=len(all_tasks),
            desc="Extracting",
            unit="vid",
            dynamic_ncols=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} vids"
                " [{elapsed}<{remaining}, {rate_fmt}]{postfix}"
            ),
        ) as pbar:
            for fut in as_completed(futures):
                try:
                    name, label, count = fut.result()
                except Exception as exc:
                    task = futures[fut]
                    tqdm.write(f"[ERROR] {task[0].name}: {exc}")
                    count, label = 0, futures[fut][2]

                if label == "real":
                    real_saved += count
                else:
                    fake_saved += count

                pbar.update(1)
                pbar.set_postfix(
                    crops=real_saved + fake_saved,
                    real=real_saved,
                    fake=fake_saved,
                )

    return real_saved, fake_saved


# ---------------------------------------------------------------------------
# Balance checker
# ---------------------------------------------------------------------------

def check_balance(real_dir: Path, fake_dir: Path, min_ratio: float = 0.8) -> None:
    real_count = len(list(real_dir.glob("*.jpg")))
    fake_count = len(list(fake_dir.glob("*.jpg")))
    total = real_count + fake_count

    log.info("─" * 52)
    log.info("Balance check")
    log.info("  real crops : %d", real_count)
    log.info("  fake crops : %d", fake_count)
    log.info("  total      : %d", total)

    if total == 0:
        log.warning("No crops found — did the pipeline run successfully?")
        return

    minority = min(real_count, fake_count)
    majority = max(real_count, fake_count)
    ratio = minority / majority if majority > 0 else 0.0
    label = "real" if real_count < fake_count else "fake"

    log.info("  ratio (minority/majority) : %.3f", ratio)

    if ratio < min_ratio:
        log.warning(
            "Class imbalance detected! %s is the minority class (ratio=%.3f < %.2f). "
            "Consider over-sampling the minority or under-sampling the majority.",
            label, ratio, min_ratio,
        )
    else:
        log.info("Balance OK (ratio=%.3f ≥ %.2f)", ratio, min_ratio)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract face crops from DFDC-style real/fake video directories."
    )
    p.add_argument("--real_dir",    required=True,  type=Path)
    p.add_argument("--fake_dir",    required=True,  type=Path)
    p.add_argument("--out_real",    required=True,  type=Path)
    p.add_argument("--out_fake",    required=True,  type=Path)
    p.add_argument("--frame_step",  default=10,          type=int)
    p.add_argument("--conf_thresh", default=0.95,        type=float)
    p.add_argument("--padding",     default=0.20,        type=float)
    p.add_argument("--workers",     default=os.cpu_count(), type=int)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--min_ratio",   default=0.8,         type=float)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    for attr, path in [("--real_dir", args.real_dir), ("--fake_dir", args.fake_dir)]:
        if not path.exists():
            log.error("%s does not exist: %s", attr, path)
            sys.exit(1)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but not available — falling back to CPU")
        device = "cpu"

    log.info("DeepSentinel preprocessing pipeline")
    log.info("  frame_step  = %d", args.frame_step)
    log.info("  conf_thresh = %.2f", args.conf_thresh)
    log.info("  padding     = %.0f%%", args.padding * 100)
    log.info("  device      = %s", device)
    log.info("  workers     = %d", args.workers)

    real_tasks = collect_tasks(
        args.real_dir, args.out_real, "real",
        args.frame_step, args.conf_thresh, args.padding, device,
    )
    fake_tasks = collect_tasks(
        args.fake_dir, args.out_fake, "fake",
        args.frame_step, args.conf_thresh, args.padding, device,
    )

    total_videos = len(real_tasks) + len(fake_tasks)
    if total_videos == 0:
        log.error("No videos found in either input directory — exiting.")
        sys.exit(1)

    log.info("Dispatching %d videos across %d workers…", total_videos, args.workers)
    real_saved, fake_saved = run_parallel(real_tasks + fake_tasks, args.workers)

    log.info("Extraction complete — real: %d  fake: %d", real_saved, fake_saved)
    check_balance(args.out_real, args.out_fake, args.min_ratio)


if __name__ == "__main__":
    sys.argv = [
        "preprocess.py",
        "--real_dir", "/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23/original",
        "--fake_dir", "/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23/Deepfakes",
        "--out_real", "/kaggle/working/crops/real",
        "--out_fake", "/kaggle/working/crops/fake"
    ]
    main()