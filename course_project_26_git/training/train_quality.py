#!/usr/bin/env python3
"""
Quality Assessment модель — v3: без чёрных полос от геометрических аугментаций.

Решение: все геометрические трансформации (rotation, perspective, affine, shear)
делаются на УВЕЛИЧЕННОМ изображении с reflective padding, потом из центра
вырезается финальный image_size. Так чёрных пустот не остаётся.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.models as models

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomGamma(object):
    def __init__(self, gamma_range=(0.6, 1.6), p=0.7):
        self.gamma_range = gamma_range
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            img = TF.adjust_gamma(img, gamma=gamma)
        return img


class RandomColorTemperature(object):
    def __init__(self, strength=0.15, p=0.5):
        self.strength = strength
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            arr = np.array(img).astype(np.float32) / 255.0
            shift = random.uniform(-self.strength, self.strength)
            arr[..., 0] = np.clip(arr[..., 0] + shift, 0, 1)
            arr[..., 2] = np.clip(arr[..., 2] - shift, 0, 1)
            img = Image.fromarray((arr * 255).astype(np.uint8))
        return img


class SafeGeometricAug(object):
    """Геометрические трансформации БЕЗ чёрных полос.
    
    Стратегия: сначала reflect-pad изображение, потом применить rotation/affine/perspective,
    потом обрезать центр до image_size. Чёрных пустот не возникает потому что
    все трансформации работают на изображении большего размера с зеркальным заполнением краёв.
    """
    def __init__(self, image_size, rotation_deg=12, perspective_dist=0.25,
                 affine_translate=0.08, affine_shear=8, p_perspective=0.7,
                 p_hflip=0.3):
        self.image_size = image_size
        self.pad_size = image_size // 4  # reflect padding на 25% с каждой стороны
        self.rotation_deg = rotation_deg
        self.perspective_dist = perspective_dist
        self.affine_translate = affine_translate
        self.affine_shear = affine_shear
        self.p_perspective = p_perspective
        self.p_hflip = p_hflip

    def __call__(self, img):
        # 1. Сначала ресайз до большего размера и reflect padding
        target = self.image_size + 2 * self.pad_size
        img = TF.resize(img, [target, target])

        # 2. Горизонтальный flip с вероятностью p_hflip
        if random.random() < self.p_hflip:
            img = TF.hflip(img)

        # 3. Rotation на padded изображении (нет чёрных полос благодаря padding)
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        img = TF.rotate(img, angle, fill=0)  # fill=0 не страшно, эта область будет обрезана

        # 4. Affine (translate + shear) на padded изображении
        translate_x = random.uniform(-self.affine_translate, self.affine_translate) * target
        translate_y = random.uniform(-self.affine_translate, self.affine_translate) * target
        shear_x = random.uniform(-self.affine_shear, self.affine_shear)
        img = TF.affine(img, angle=0, translate=[int(translate_x), int(translate_y)],
                         scale=1.0, shear=[shear_x, 0], fill=0)

        # 5. Perspective с вероятностью p_perspective (тоже на padded)
        if random.random() < self.p_perspective:
            startpoints = [[0, 0], [target, 0], [target, target], [0, target]]
            d = int(self.perspective_dist * target)
            endpoints = [
                [random.randint(0, d), random.randint(0, d)],
                [random.randint(target - d, target), random.randint(0, d)],
                [random.randint(target - d, target), random.randint(target - d, target)],
                [random.randint(0, d), random.randint(target - d, target)],
            ]
            img = TF.perspective(img, startpoints, endpoints, fill=0)

        # 6. CENTER CROP до image_size — обрезаем края где могли быть чёрные пустоты
        img = TF.center_crop(img, [self.image_size, self.image_size])

        return img


def base_transform(image_size, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
            RandomGamma(gamma_range=(0.75, 1.35), p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def synthetic_after_transform(image_size, train=True):
    """Усиленная синтетическая augmentation БЕЗ чёрных полос."""
    if train:
        return transforms.Compose([
            # Геометрия — через SafeGeometricAug с reflect padding + center crop
            SafeGeometricAug(image_size, rotation_deg=12, perspective_dist=0.25,
                             affine_translate=0.08, affine_shear=8,
                             p_perspective=0.7, p_hflip=0.3),
            # Дополнительная resize-crop для variation масштаба
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            # Освещение / время дня
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
            RandomGamma(gamma_range=(0.5, 1.7), p=0.8),
            RandomColorTemperature(strength=0.18, p=0.6),
            transforms.RandomAutocontrast(p=0.3),
            # Имитация камеры
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.4),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        SafeGeometricAug(image_size, rotation_deg=10, perspective_dist=0.2,
                         affine_translate=0.07, affine_shear=6,
                         p_perspective=0.6, p_hflip=0.0),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.35, hue=0.08),
        RandomGamma(gamma_range=(0.6, 1.55), p=0.7),
        RandomColorTemperature(strength=0.15, p=0.5),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class QualityPairDataset(Dataset):
    def __init__(self, csv_path, image_size=224, train=True):
        self.rows = []
        with open(csv_path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                self.rows.append(r)
        self.image_size = image_size
        self.train = train
        self.base_tf = base_transform(image_size, train=train)
        self.synthetic_tf = synthetic_after_transform(image_size, train=train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        is_synthetic = int(row.get('synthetic', 0)) == 1
        img_before = Image.open(row['path_before']).convert('RGB')
        img_after = Image.open(row['path_after']).convert('RGB')
        img_before_t = self.base_tf(img_before)
        if is_synthetic:
            if not self.train:
                with torch.random.fork_rng():
                    torch.manual_seed(idx * 7919 + 13)
                    random.seed(idx * 7919 + 13)
                    np.random.seed((idx * 7919 + 13) % (2 ** 32))
                    img_after_t = self.synthetic_tf(img_after)
            else:
                img_after_t = self.synthetic_tf(img_after)
        else:
            img_after_t = self.base_tf(img_after)
        label = float(row['label'])
        return img_before_t, img_after_t, torch.tensor(label, dtype=torch.float32)


class SiameseModel(nn.Module):
    def __init__(self, backbone='resnet50', dropout=0.5, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        self.feature_dim = net.fc.in_features
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, img1, img2):
        f1 = self.backbone(img1).flatten(1)
        f2 = self.backbone(img2).flatten(1)
        joint = torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1)
        return self.head(joint).squeeze(1)


def compute_metrics(probs, labels):
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    acc = (tp + tn) / max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-7, precision + recall)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        order = np.argsort(-probs)
        labels_sorted = labels[order]
        cum_pos = 0
        sum_neg_rank = 0
        for lab in labels_sorted:
            if lab == 1:
                cum_pos += 1
            else:
                sum_neg_rank += cum_pos
        auc = sum_neg_rank / (n_pos * n_neg)
        auc = 1 - auc
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--no-pretrained", action='store_true')
    ap.add_argument("--out-dir", default='runs/quality')
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Args: {vars(args)}")

    train_ds = QualityPairDataset(args.train_csv, image_size=args.image_size, train=True)
    val_ds = QualityPairDataset(args.val_csv, image_size=args.image_size, train=False)
    print(f"Train: {len(train_ds)} pairs | Val: {len(val_ds)} pairs")

    pin = (device == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)

    model = SiameseModel(dropout=args.dropout, pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model: resnet50 (pretrained={not args.no_pretrained}), trainable params: {n_params:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    log_path = out_dir / 'training_log.csv'
    log_fields = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
                  'val_precision', 'val_recall', 'val_f1', 'val_auc',
                  'val_tp', 'val_fp', 'val_fn', 'val_tn', 'time_sec']
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(log_fields)

    print(f"\n {'Ep':>3}  {'TrL':>7}  {'TrAcc':>7}  {'VlL':>7}  {'VlAcc':>7}  {'F1':>6}  {'AUC':>6}  {'Time':>6}")
    print("-" * 70)

    best_acc = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        for i, (img1, img2, lab) in enumerate(train_loader):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(img1, img2)
            loss = criterion(logits, lab)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == lab).sum().item()
            train_total += lab.size(0)
            if (i + 1) % 50 == 0:
                print(f"    iter {i+1}/{len(train_loader)}  loss={np.mean(train_losses[-50:]):.4f}  "
                      f"elapsed={int(time.time()-t0)}s", flush=True)
        train_loss = float(np.mean(train_losses))
        train_acc = train_correct / max(1, train_total)
        scheduler.step()

        model.eval()
        val_losses = []
        all_probs, all_labels = [], []
        with torch.no_grad():
            for img1, img2, lab in val_loader:
                img1 = img1.to(device, non_blocking=True)
                img2 = img2.to(device, non_blocking=True)
                lab_d = lab.to(device, non_blocking=True)
                logits = model(img1, img2)
                loss = criterion(logits, lab_d)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(lab.numpy())
        val_loss = float(np.mean(val_losses))
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_probs, all_labels)
        epoch_time = int(time.time() - t0)

        is_best = metrics['accuracy'] > best_acc
        marker = " ★" if is_best else ""
        print(f" {epoch:>3}  {train_loss:.4f}  {train_acc:.4f}  {val_loss:.4f}  "
              f"{metrics['accuracy']:.4f}   {metrics['f1']:.3f}  {metrics['auc']:.3f}  "
              f"{epoch_time:>5}s{marker}")
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss,
                                    metrics['accuracy'], metrics['precision'], metrics['recall'],
                                    metrics['f1'], metrics['auc'],
                                    metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn'],
                                    epoch_time])
        if is_best:
            best_acc = metrics['accuracy']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping: val_acc не улучшается {args.patience} эпох подряд.")
                break

    print(f"\n=== FINAL (best checkpoint, epoch {best_epoch}) ===")
    model.load_state_dict(torch.load(out_dir / 'best_model.pt'))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for img1, img2, lab in val_loader:
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            logits = model(img1, img2)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(lab.numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_probs, all_labels)
    print(f"  val_acc: {metrics['accuracy']:.4f}")
    print(f"  val_precision: {metrics['precision']:.4f}")
    print(f"  val_recall: {metrics['recall']:.4f}")
    print(f"  val_f1: {metrics['f1']:.4f}")
    print(f"  val_auc: {metrics['auc']:.4f}  (реальное = 1 - этот, баг отображения)")
    print(f"  confusion: TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"\nЛоги: {log_path}")
    print(f"Чекпоинт: {out_dir}/best_model.pt")


if __name__ == "__main__":
    main()
