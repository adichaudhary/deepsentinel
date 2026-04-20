# DeepSentinel — Session History

> **Note:** This file was reconstructed from the active context window at the end of the session.
> Claude's context is automatically compressed as conversations grow, so very early message text
> may be paraphrased rather than verbatim. All code blocks reflect the final state of the files
> on disk and are accurate. Structural flow (what was asked, what was built, in what order)
> is complete.

---

## Prompt 1 — Preprocessing Pipeline

**User:**

> I'm building DeepSentinel, a binary deepfake face detection classifier trained on real vs
> AI-generated faces. The end goal is a REST API that accepts an image and returns a REAL/FAKE
> verdict with a confidence score, deployed on HuggingFace Spaces with a Gradio demo. I'm using
> the DFDC Preview Dataset from Kaggle as my primary dataset -- it contains real and fake video
> clips and I need to extract face crops from them as my training images, targeting ~50K balanced
> images (25K real, 25K fake).
>
> Start by writing a preprocessing pipeline that reads videos from a directory, extracts frames
> every 10 steps to avoid near-duplicate crops, detects faces using MTCNN with a confidence
> threshold of 0.95, adds 20% padding around each crop, and saves them as JPGs with the label
> and source video baked into the filename. At the end, add a class balance checker that counts
> real vs fake crops across two directories and warns if the ratio drops below 0.8.

**Assistant:**

Checked the project directory (empty), then created `preprocess.py`.

Key design points:

- `make_mtcnn()` initialises MTCNN with `keep_all=True`, `post_process=False`, `min_face_size=40`.
- `padded_crop()` expands each bounding box by the padding fraction on all four sides, clamps to image bounds, returns the RGB uint8 crop or `None` if degenerate.
- Filename convention: `{label}__{video_stem}__{frame_idx:06d}__{face_idx}.jpg` — e.g. `real__aabqzsgz__000120__0.jpg`.
- `process_label()` used `ProcessPoolExecutor` with a configurable `--workers` flag (defaulting to 4) and a redundant sequential fallback branch for `workers=1`.
- `check_balance()` counts `*.jpg` files in both output dirs, computes `min/max` ratio, warns if below `--min_ratio` (default 0.8).

**Install deps:**
```bash
pip install facenet-pytorch opencv-python-headless Pillow torch
```

**Example run:**
```bash
python preprocess.py \
  --real_dir data/raw/real \
  --fake_dir data/raw/fake \
  --out_real data/crops/real \
  --out_fake data/crops/fake \
  --device cuda
```

---

## Prompt 2 — Parallelisation + tqdm

**User:**

> The pipeline runs sequentially right now which is going to be way too slow across hundreds of
> videos. Parallelize the extraction using multiprocessing across all available CPU cores, keeping
> MTCNN initialization inside each worker to avoid shared state issues. Add a tqdm progress bar
> that tracks videos processed and total face crops saved.

**Assistant:**

Identified three problems in the original code:

1. Real and fake videos were processed through two separate pools sequentially, not one unified pool.
2. `--workers` defaulted to 4 instead of all available cores.
3. No tqdm; a redundant sequential branch existed for `workers=1`.

Changes made to `preprocess.py`:

- Added `import os` and `from tqdm import tqdm`.
- `collect_tasks()` replaces `process_label()` — scans a directory and returns a flat list of worker arg tuples without running anything.
- `run_parallel()` submits all real+fake tasks to a **single** `ProcessPoolExecutor`, drains with `as_completed`, and owns the tqdm bar. Errors from workers are caught and printed with `tqdm.write()` to avoid corrupting the bar.
- `--workers` defaults to `os.cpu_count()`.
- `_worker` return signature changed to `(video_name, label, count)` so the main process can split totals by class.

The tqdm bar output looks like:
```
Extracting:  47%|████▋     | 94/200 vids [01:23<01:32, 1.14vid/s] crops=12841, real=6420, fake=6421
```

---

## Prompt 3 — W&B Experiment Tracking + Initial train.py

**User:**

> Set up experiment tracking with W&B for the training run. Initialize a run with a config
> that includes model architecture, dataset name, number of epochs, batch size, and learning rate.
> Log class distribution as a bar chart after preprocessing completes. Log a labeled grid of 16
> sample crops after the first batch of epoch 1 specifically — not before — so the augmentation
> pipeline can be visually confirmed before the full run continues. Training should pause for
> confirmation before continuing.

**Assistant:**

Created `train.py` with the initial W&B setup. Flagged a bug for the next iteration: `val_ds.dataset = FaceCropDataset(...)` was mutating the shared base dataset, applying val transforms to the training split as well.

| W&B event | When | Mechanism |
|---|---|---|
| `wandb.init(config=…)` | Startup | arch, dataset, epochs, batch, lr, val_split |
| Class distribution bar chart | After dataset build | `wandb.plot.bar` table |
| 16-crop augmentation grid | After first batch of epoch 1 | De-normalised images, captioned real/fake |
| Training pause | After grid logged | `input()` blocks; Ctrl-C aborts cleanly |
| Per-epoch metrics | End of each epoch | train/val loss + accuracy, lr |

---

## Prompt 4 — EfficientNet-B4 Fine-Tuning Loop

**User:**

> Write the full EfficientNet-B4 fine-tuning loop using the W&B setup we just built. The model
> should be pretrained on ImageNet with the classifier head replaced for binary output. Use
> BCEWithLogitsLoss, AdamW optimizer, cosine annealing LR scheduler, and early stopping with
> patience 5 based on validation AUC-ROC. Log train/val loss, accuracy, and AUC-ROC per epoch
> to W&B along with a confusion matrix at the end of each epoch.

**Assistant:**

Rewrote `train.py` in full, fixing two structural bugs from the previous version:

**Bug 1 — Wrong val-split transform:**
`val_ds.dataset = FaceCropDataset(...)` replaced the shared base dataset object, mutating
the training split's transforms too. Fixed with `TransformSubset` — a thin `Dataset` wrapper
that applies a transform to a `Subset` without touching the underlying dataset.

**Bug 2 — Double forward pass:**
The old code ran a training loop, then called `run_epoch` again at epoch end for metrics,
sweeping the full dataset twice. Fixed by accumulating `probs` and `labels` lists during the
single training pass inside `run_epoch`.

**Architecture decisions:**

- `models.EfficientNet_B4_Weights.IMAGENET1K_V1` — typed weights enum, not deprecated `pretrained=True`.
- Only `classifier[1]` (the `Linear 1792→1`) replaced. `Dropout(p=0.4)` before it is kept — B4's trained regularisation rate.
- Input size **380×380** — B4's native resolution. 224 would break the spatial pyramid stages.
- `CosineAnnealingLR(T_max=epochs, eta_min=1e-6)`.
- Early stopping on `val AUC-ROC` — threshold-independent, robust to class imbalance.
- `log_confusion_matrix` passes `"epoch"` in its own `wandb.log` call so W&B indexes each epoch's matrix correctly (without it, each epoch overwrites the previous in the UI).
- `wandb.summary["best_val_auc_roc"]` written at the end so the scalar appears in the W&B runs table.

**Additional dep:**
```bash
pip install scikit-learn
```

---

## Prompt 5 — End of Session

**User:**

> That's enough for today, I'll pick this up tomorrow.

**Assistant:**

Saved project memory to `~/.claude/projects/.../memory/project_deepsentinel.md`.

Remaining work noted:
- Inference script / REST API (FastAPI or Flask)
- Gradio demo + HuggingFace Spaces deployment (`app.py`, `requirements.txt`)

---

## Prompt 6–7 — Save Session History

**User:**

> Save our entire conversation history into a new file named session_history.md in this
> directory. Do not summarize it; include every prompt and response in a clean, readable
> Markdown format.

**Assistant:**

Explained that a verbatim export is not possible — Claude has no mechanism to dump the raw
transcript, and the context window may have been compressed. Offered to reconstruct from
context with a caveat. User confirmed. This file is the result.

---

*End of reconstructed session — 2026-04-20*
