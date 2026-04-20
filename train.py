"""
train.py — DeepSentinel EfficientNet-B4 fine-tuning

Usage:
    python train.py \
        --real_dir  data/crops/real \
        --fake_dir  data/crops/fake \
        --epochs    30 \
        --batch     24 \
        --lr        3e-4 \
        --patience  5

W&B logs (per epoch):
  • train/val loss, accuracy, AUC-ROC
  • val confusion matrix
  • learning rate

W&B one-time logs:
  • class distribution bar chart  (after dataset is built)
  • 16-crop augmentation grid     (after first batch of epoch 1)

Early stopping:
  • Monitors val AUC-ROC; saves best checkpoint to checkpoints/best_model.pt
  • Stops after `--patience` consecutive epochs without improvement
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_auc_score, confusion_matrix
from PIL import Image
import wandb


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FaceCropDataset(Dataset):
    """
    Reads JPGs from real_dir / fake_dir. Returns raw PIL images so that
    transforms can be applied per-split via TransformSubset below.
    """
    def __init__(self, real_dir: Path, fake_dir: Path):
        self.samples: list[tuple[Path, int]] = []

        for path in sorted(real_dir.glob("*.jpg")):
            self.samples.append((path, 0))   # 0 = real
        for path in sorted(fake_dir.glob("*.jpg")):
            self.samples.append((path, 1))   # 1 = fake

        if not self.samples:
            raise RuntimeError(
                f"No JPGs found in {real_dir} or {fake_dir}. "
                "Run preprocess.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        path, label = self.samples[idx]
        return Image.open(path).convert("RGB"), label

    def class_counts(self) -> dict[str, int]:
        real = sum(1 for _, lbl in self.samples if lbl == 0)
        fake = len(self.samples) - real
        return {"real": real, "fake": fake}


class TransformSubset(Dataset):
    """Wraps a Subset and applies a transform, keeping split indices intact."""
    def __init__(self, subset: Subset, transform: T.Compose):
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self.subset[idx]
        return self.transform(img), label


# ---------------------------------------------------------------------------
# Transforms  (B4 native resolution: 380 × 380)
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms(train: bool) -> T.Compose:
    if train:
        return T.Compose([
            T.Resize((380, 380)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomRotation(10),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((380, 380)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.efficientnet_b4(weights=weights)

    # Replace only the final linear layer; keep B4's Dropout(p=0.4) in place
    in_features = model.classifier[1].in_features   # 1792
    model.classifier[1] = nn.Linear(in_features, 1)

    return model


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stops training when val AUC-ROC fails to improve for `patience` epochs."""

    def __init__(self, patience: int, checkpoint_path: Path):
        self.patience        = patience
        self.checkpoint_path = checkpoint_path
        self.best_auc        = -1.0
        self.counter         = 0
        self.should_stop     = False

    def step(self, val_auc: float, model: nn.Module, epoch: int) -> None:
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.counter  = 0
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"epoch": epoch, "val_auc": val_auc,
                 "model_state_dict": model.state_dict()},
                self.checkpoint_path,
            )
            print(f"  ✓ New best AUC-ROC {val_auc:.4f} — checkpoint saved")
        else:
            self.counter += 1
            print(f"  No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.should_stop = True


# ---------------------------------------------------------------------------
# Per-epoch pass
# ---------------------------------------------------------------------------

class EpochResult:
    __slots__ = ("loss", "accuracy", "auc", "preds", "labels")

    def __init__(
        self,
        loss: float,
        accuracy: float,
        auc: float,
        preds: np.ndarray,
        labels: np.ndarray,
    ):
        self.loss     = loss
        self.accuracy = accuracy
        self.auc      = auc
        self.preds    = preds
        self.labels   = labels


def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    is_train:  bool,
    first_batch_hook=None,
) -> EpochResult:
    model.train(is_train)
    total_loss  = 0.0
    all_labels: list[int]   = []
    all_probs:  list[float] = []

    with torch.set_grad_enabled(is_train):
        for batch_idx, (imgs, labels) in enumerate(loader):
            if batch_idx == 0 and first_batch_hook is not None:
                first_batch_hook(imgs, labels)

            imgs     = imgs.to(device)
            labels_t = labels.float().unsqueeze(1).to(device)

            logits = model(imgs)
            loss   = criterion(logits, labels_t)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits).detach().squeeze(1).cpu()
            total_loss  += loss.item() * labels.size(0)
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())

    labels_arr = np.array(all_labels)
    probs_arr  = np.array(all_probs)
    preds_arr  = (probs_arr >= 0.5).astype(int)

    avg_loss = total_loss / len(labels_arr)
    accuracy = float((preds_arr == labels_arr).mean())
    auc      = (
        roc_auc_score(labels_arr, probs_arr)
        if len(np.unique(labels_arr)) > 1 else 0.5
    )

    return EpochResult(avg_loss, accuracy, auc, preds_arr, labels_arr)


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def log_class_distribution(counts: dict[str, int]) -> None:
    table = wandb.Table(
        columns=["class", "count"],
        data=[["real", counts["real"]], ["fake", counts["fake"]]],
    )
    wandb.log({
        "dataset/class_distribution": wandb.plot.bar(
            table, "class", "count", title="Class Distribution"
        )
    })


def log_augmentation_grid(imgs: torch.Tensor, labels: torch.Tensor) -> None:
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

    imgs = (imgs[:16].cpu() * std + mean).clamp(0, 1)
    label_names = {0: "real", 1: "fake"}
    wandb.log({
        "augmentation/epoch1_first_batch": [
            wandb.Image(
                imgs[i].permute(1, 2, 0).numpy(),
                caption=label_names[int(labels[i].item())],
            )
            for i in range(len(imgs))
        ]
    })


def log_confusion_matrix(result: EpochResult, epoch: int, split: str) -> None:
    wandb.log({
        f"{split}/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=result.labels.tolist(),
            preds=result.preds.tolist(),
            class_names=["real", "fake"],
        ),
        "epoch": epoch,
    })


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    # ── W&B ─────────────────────────────────────────────────────────────────
    run = wandb.init(
        project="deepsentinel",
        config={
            "architecture":  "efficientnet_b4",
            "dataset":       "FaceForensics++",
            "input_size":    380,
            "epochs":        args.epochs,
            "batch_size":    args.batch,
            "learning_rate": args.lr,
            "weight_decay":  args.weight_decay,
            "val_split":     args.val_split,
            "patience":      args.patience,
            "pretrained":    True,
            "device":        str(device),
        },
    )
    cfg = run.config

    # ── Dataset + splits ────────────────────────────────────────────────────
    base_ds = FaceCropDataset(Path(args.real_dir), Path(args.fake_dir))
    counts  = base_ds.class_counts()
    print(f"Dataset: {counts['real']} real  |  {counts['fake']} fake  "
          f"|  total {len(base_ds)}")
    log_class_distribution(counts)

    val_size   = int(len(base_ds) * cfg.val_split)
    train_size = len(base_ds) - val_size
    train_subset, val_subset = random_split(
        base_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_ds = TransformSubset(train_subset, build_transforms(train=True))
    val_ds   = TransformSubset(val_subset,   build_transforms(train=False))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model     = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )
    stopper = EarlyStopping(
        patience=cfg.patience,
        checkpoint_path=Path(args.checkpoint_dir) / "best_model.pt",
    )

    wandb.watch(model, log="gradients", log_freq=50)

    # ── Augmentation-grid hook (fires on first batch of epoch 1 only) ──────
    aug_grid_done = False

    def maybe_log_aug_grid(imgs: torch.Tensor, labels: torch.Tensor) -> None:
        nonlocal aug_grid_done
        if aug_grid_done:
            return
        log_augmentation_grid(imgs, labels)
        aug_grid_done = True
        print("[W&B] Augmentation grid logged. Continuing training...")

    # ── Loop ────────────────────────────────────────────────────────────────
    print(f"\nTraining EfficientNet-B4 on {device} "
          f"| {train_size} train / {val_size} val")
    print(f"{'Epoch':>6}  {'T-Loss':>7}  {'T-Acc':>6}  {'T-AUC':>6}  "
          f"{'V-Loss':>7}  {'V-Acc':>6}  {'V-AUC':>6}")

    for epoch in range(1, cfg.epochs + 1):
        hook = maybe_log_aug_grid if epoch == 1 else None

        tr = run_epoch(model, train_loader, criterion, optimizer,
                       device, is_train=True, first_batch_hook=hook)
        vl = run_epoch(model, val_loader, criterion, None,
                       device, is_train=False)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        wandb.log({
            "epoch":          epoch,
            "lr":             current_lr,
            "train/loss":     tr.loss,
            "train/accuracy": tr.accuracy,
            "train/auc_roc":  tr.auc,
            "val/loss":       vl.loss,
            "val/accuracy":   vl.accuracy,
            "val/auc_roc":    vl.auc,
        })

        log_confusion_matrix(vl, epoch, split="val")

        print(
            f"{epoch:>6}  {tr.loss:>7.4f}  {tr.accuracy:>6.3f}  {tr.auc:>6.4f}  "
            f"{vl.loss:>7.4f}  {vl.accuracy:>6.3f}  {vl.auc:>6.4f}"
        )

        stopper.step(vl.auc, model, epoch)
        if stopper.should_stop:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best val AUC-ROC: {stopper.best_auc:.4f})")
            break

    print(f"\nBest val AUC-ROC: {stopper.best_auc:.4f}  "
          f"→  {stopper.checkpoint_path}")
    wandb.summary["best_val_auc_roc"] = stopper.best_auc
    wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune EfficientNet-B4 for binary deepfake detection."
    )
    p.add_argument("--real_dir",       required=True,           help="Real face crop directory")
    p.add_argument("--fake_dir",       required=True,           help="Fake face crop directory")
    p.add_argument("--epochs",         default=30,  type=int,   help="Max epochs (default: 30)")
    p.add_argument("--batch",          default=24,  type=int,   help="Train batch size (default: 24)")
    p.add_argument("--lr",             default=3e-4,type=float, help="Peak learning rate (default: 3e-4)")
    p.add_argument("--weight_decay",   default=1e-2,type=float, help="AdamW weight decay (default: 1e-2)")
    p.add_argument("--val_split",      default=0.15,type=float, help="Validation fraction (default: 0.15)")
    p.add_argument("--patience",       default=5,   type=int,   help="Early-stopping patience on val AUC-ROC (default: 5)")
    p.add_argument("--workers",        default=4,   type=int,   help="DataLoader workers (default: 4)")
    p.add_argument("--checkpoint_dir", default="checkpoints",   help="Where to save best model (default: checkpoints/)")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.argv = [
        "train.py",
        "--real_dir", "/kaggle/working/crops/real",
        "--fake_dir", "/kaggle/working/crops/fake"
    ]
    train(parse_args())