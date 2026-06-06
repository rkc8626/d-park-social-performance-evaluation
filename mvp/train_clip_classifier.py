#!/usr/bin/env python3
"""Train activity + age classifiers on labeled clip JPEGs (multi-frame avg logits)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from export_track_clips import ACTIVITIES, AGE_GROUPS


class ClipDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str,
        classes: list[str],
        image_size: int,
        augment: bool,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if augment:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                    transforms.ToTensor(),
                    norm,
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    norm,
                ]
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        batch_dir = Path(row["batch_dir"])
        clip_dir = batch_dir / row["clip_subdir"]
        frames = sorted(clip_dir.glob("frame_*.jpg"))
        if not frames:
            raise FileNotFoundError(clip_dir)
        tensors = []
        for fp in frames[:5]:
            img = Image.open(fp).convert("RGB")
            tensors.append(self.tf(img))
        # (N, 3, H, W) -> average logits at train time in loop; here stack for batch
        x = torch.stack(tensors, dim=0)
        y = self.class_to_idx[str(row[self.label_col])]
        return x, y


def _collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def _make_model(n_classes: int) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            # x: (B, F, 3, H, W)
            b, f, c, h, w = x.shape
            x = x.to(device)
            y = y.to(device)
            logits_sum = None
            for fi in range(f):
                out = model(x[:, fi])
                logits_sum = out if logits_sum is None else logits_sum + out
            logits = logits_sum / f
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * b
            correct += int((logits.argmax(1) == y).sum().item())
            n += b
    return total_loss / max(n, 1), correct / max(n, 1)


def train_task(
    name: str,
    df: pd.DataFrame,
    label_col: str,
    classes: list[str],
    models_dir: Path,
    cfg: dict,
    device: torch.device,
) -> dict:
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    if val_df.empty:
        val_df = train_df.sample(min(5, len(train_df)), random_state=cfg.get("seed", 42))

    image_size = int(cfg.get("image_size", 224))
    train_ds = ClipDataset(train_df, label_col, classes, image_size, augment=True)
    val_ds = ClipDataset(val_df, label_col, classes, image_size, augment=False)
    nw = int(cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=nw,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=nw,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )

    model = _make_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    best_acc = 0.0
    if name == "activity":
        best_path = models_dir / cfg.get("activity_model", "activity_resnet18.pt")
    else:
        best_path = models_dir / cfg.get("age_model", "age_resnet18.pt")

    epochs = int(cfg.get("epochs", 40))
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, None, device)
        print(
            f"[{name}] ep {ep}/{epochs} train loss={tr_loss:.4f} acc={tr_acc:.3f} "
            f"val loss={va_loss:.4f} acc={va_acc:.3f}"
        )
        if va_acc >= best_acc:
            best_acc = va_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": classes,
                    "label_col": label_col,
                    "image_size": image_size,
                },
                best_path,
            )
    print(f"[{name}] best val acc={best_acc:.3f} -> {best_path}")
    return {"classes": classes, "best_val_acc": best_acc, "path": str(best_path)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("classifier_config.yaml"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text())
    index_path = (root / cfg["dataset_index"]).resolve()
    if not index_path.is_file():
        raise SystemExit(f"Run build_clip_dataset_index.py first. Missing {index_path}")

    df = pd.read_csv(index_path)
    models_dir = (Path(__file__).parent / cfg.get("models_dir", "models")).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print("Device:", device, "| clips:", len(df))

    meta: dict = {}
    meta["activity"] = train_task("activity", df, "activity_label", ACTIVITIES, models_dir, cfg, device)
    meta["age"] = train_task("age", df, "apparent_age_group", AGE_GROUPS, models_dir, cfg, device)

    labels_path = models_dir / cfg.get("labels_json", "classifier_labels.json")
    labels_path.write_text(json.dumps(meta, indent=2))
    print("Wrote", labels_path)


if __name__ == "__main__":
    main()
