#!/usr/bin/env python3
"""
Siamese network для Location Verification (модуль 1 из proposed architecture).
Улучшенный baseline vs ResNet18-from-scratch: pretrained ResNet50 + augmentation
+ AdamW + cosine schedule.

CSV формат (для --train-csv и --val-csv):
    path_before,path_after,label
    /full/path/to/before1.jpg,/full/path/to/after1.jpg,1
    ...
    label: 1 = same location (original pair), 0 = different location (negative)

Запуск:
    python train_siamese.py --train-csv train.csv --val-csv val.csv --epochs 10

На Mac (M1/M2/M3) автоматически использует MPS. На Linux/Windows с GPU — CUDA.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from PIL import Image, ImageFile

# на случай частично битых фото
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------- Dataset ---------------------------

class PairDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.rows = []
        with open(csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rows.append(r)
        self.transform = transform
        # на проблемных файлах вернём None, далее отфильтруем в collate_fn
        self._bad = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        try:
            img_b = Image.open(r["path_before"]).convert("RGB")
            img_a = Image.open(r["path_after"]).convert("RGB")
        except Exception:
            self._bad += 1
            return None
        if self.transform:
            img_b = self.transform(img_b)
            img_a = self.transform(img_a)
        label = float(r["label"])
        return img_b, img_a, torch.tensor(label, dtype=torch.float32)


def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    img_b = torch.stack([x[0] for x in batch])
    img_a = torch.stack([x[1] for x in batch])
    labels = torch.stack([x[2] for x in batch])
    return img_b, img_a, labels


# --------------------------- Model ---------------------------

class SiameseResNet(nn.Module):
    """
    Shared backbone (ResNet) → embed obe фото независимо →
    combine(concat, abs-diff) → MLP-голова → logit.
    """
    def __init__(self, backbone="resnet50", pretrained=True, freeze_backbone=False,
                 dropout=0.5, hidden=512):
        super().__init__()
        if backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            feat_dim = 2048
        elif backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # вход головы: concat(f_b, f_a, |f_b - f_a|) → 3 * feat_dim
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, img_b, img_a):
        f_b = self.backbone(img_b)
        f_a = self.backbone(img_a)
        combined = torch.cat([f_b, f_a, torch.abs(f_b - f_a)], dim=1)
        return self.head(combined).squeeze(1)


# --------------------------- Metrics ---------------------------

def compute_metrics(probs, labels):
    """Возвращает acc, precision, recall, f1, auc."""
    import numpy as np
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    preds = (probs > 0.5).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    acc = (tp + tn) / max(1, len(labels))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    # AUC через простую формулу (без sklearn зависимости)
    order = np.argsort(-probs)
    labels_sorted = labels[order]
    n_pos = labels_sorted.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        cum_pos = 0
        sum_neg_rank = 0
        for lab in labels_sorted:
            if lab == 1:
                cum_pos += 1
            else:
                sum_neg_rank += cum_pos
        auc = sum_neg_rank / (n_pos * n_neg)
        auc = 1 - auc  # т.к. отсортированы по убыванию вероятности

    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# --------------------------- Train / Eval ---------------------------

def run_epoch(model, loader, criterion, optimizer, device, train=True, log_every=50):
    model.train() if train else model.eval()
    total_loss, total_samples = 0.0, 0
    all_probs, all_labels = [], []

    t0 = time.time()
    with torch.set_grad_enabled(train):
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            img_b, img_a, labels = batch
            img_b = img_b.to(device, non_blocking=True)
            img_a = img_a.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(img_b, img_a)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

            if train and (i + 1) % log_every == 0:
                avg_loss = total_loss / total_samples
                elapsed = time.time() - t0
                print(f"    iter {i+1}/{len(loader)}  loss={avg_loss:.4f}  "
                      f"elapsed={elapsed:.0f}s", flush=True)

    avg_loss = total_loss / max(1, total_samples)
    metrics = compute_metrics(all_probs, all_labels)
    return avg_loss, metrics


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Siamese network для Location Verification")
    ap.add_argument("--train-csv", required=True, help="CSV: path_before,path_after,label")
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="заморозить backbone (быстрее, но потолок точности ниже)")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="отключить pretrained ImageNet веса (для ablation)")
    ap.add_argument("--patience", type=int, default=3, help="early stopping patience")
    ap.add_argument("--out-dir", default="runs/siamese")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Args: {vars(args)}")

    # Augmentations
    train_tf = T.Compose([
        T.Resize((args.image_size + 32, args.image_size + 32)),
        T.RandomCrop(args.image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = PairDataset(args.train_csv, train_tf)
    val_ds = PairDataset(args.val_csv, val_tf)
    print(f"Train: {len(train_ds)} pairs | Val: {len(val_ds)} pairs")

    # pin_memory тормозит на mps — отключаем
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin_memory,
                              collate_fn=collate_skip_none, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_memory,
                            collate_fn=collate_skip_none, persistent_workers=args.workers > 0)

    model = SiameseResNet(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.backbone} (pretrained={not args.no_pretrained}), "
          f"trainable params: {n_trainable/1e6:.1f}M")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = []
    best_val_acc = 0.0
    patience_left = args.patience

    header = (f"\n{'Ep':>3}  {'TrL':>7}  {'TrAcc':>7}  {'VlL':>7}  "
              f"{'VlAcc':>7}  {'F1':>6}  {'AUC':>6}  Time")
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_m = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_m = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        dt = time.time() - t0

        marker = ""
        if val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_m,
                "args": vars(args),
            }, out_dir / "best_model.pt")
            marker = " ★"
            patience_left = args.patience
        else:
            patience_left -= 1

        print(f"{epoch:>3}  {train_loss:>7.4f}  {train_m['acc']:>7.4f}  "
              f"{val_loss:>7.4f}  {val_m['acc']:>7.4f}  {val_m['f1']:>6.3f}  "
              f"{val_m['auc']:>6.3f}  {dt:>5.0f}s{marker}")

        log.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_m["acc"],
            "val_loss": val_loss, "val_acc": val_m["acc"],
            "val_precision": val_m["precision"], "val_recall": val_m["recall"],
            "val_f1": val_m["f1"], "val_auc": val_m["auc"],
            "val_tp": val_m["tp"], "val_fp": val_m["fp"],
            "val_fn": val_m["fn"], "val_tn": val_m["tn"],
            "lr": scheduler.get_last_lr()[0], "time_sec": dt,
        })

        if patience_left <= 0:
            print(f"\nEarly stopping: val_acc не улучшается {args.patience} эпох подряд.")
            break

    # сохранить логи и финальную сводку
    with open(out_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # финальная оценка best-чекпоинта
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_m = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
    print(f"\n=== FINAL (best checkpoint, epoch {ckpt['epoch']}) ===")
    for k in ["acc", "precision", "recall", "f1", "auc"]:
        print(f"  val_{k}: {final_m[k]:.4f}")
    print(f"  confusion: TP={final_m['tp']}  FP={final_m['fp']}  "
          f"FN={final_m['fn']}  TN={final_m['tn']}")
    print(f"\nЛоги: {out_dir}/training_log.csv")
    print(f"Чекпоинт: {out_dir}/best_model.pt")


if __name__ == "__main__":
    main()
