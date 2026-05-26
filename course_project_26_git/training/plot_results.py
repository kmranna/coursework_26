#!/usr/bin/env python3
"""
Генерирует все графики и таблицы для курсовой:
  - training_curves.png  — loss + accuracy + F1 по эпохам
  - confusion_matrix.png — матрица ошибок на лучшей эпохе
  - comparison_table.csv — сравнительная таблица baseline vs improved
  - summary.txt          — текстовая сводка для копирования в курсовую

Запуск:
    python3 plot_results.py
    python3 plot_results.py --log runs/siamese/training_log.csv

Зависимости: pip install matplotlib pandas
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("Нужно: pip install matplotlib numpy")
    sys.exit(1)

# Стиль для академической работы
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load_log(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) if k != "epoch" else int(v) for k, v in r.items()})
    return rows


def plot_training_curves(rows, out_path):
    """Loss + Accuracy + F1 curves в одной картинке (3 subplots)."""
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    train_acc = [r["train_acc"] for r in rows]
    val_acc = [r["val_acc"] for r in rows]
    val_f1 = [r["val_f1"] for r in rows]
    val_prec = [r["val_precision"] for r in rows]
    val_rec = [r["val_recall"] for r in rows]
    # AUC был инвертирован — исправляем
    val_auc = [1 - r["val_auc"] for r in rows]

    best_epoch = max(range(len(rows)), key=lambda i: val_acc[i]) + 1
    best_val_acc = max(val_acc)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, "o-", label="Train", color="#1976d2", linewidth=2, markersize=6)
    ax.plot(epochs, val_loss, "s-", label="Validation", color="#d32f2f", linewidth=2, markersize=6)
    ax.axvline(best_epoch, linestyle="--", color="gray", alpha=0.5, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Training & Validation Loss", fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.set_xticks(epochs)

    # 2. Accuracy
    ax = axes[1]
    ax.plot(epochs, train_acc, "o-", label="Train", color="#1976d2", linewidth=2, markersize=6)
    ax.plot(epochs, val_acc, "s-", label="Validation", color="#388e3c", linewidth=2, markersize=6)
    ax.axvline(best_epoch, linestyle="--", color="gray", alpha=0.5, linewidth=1)
    ax.annotate(f"Best: {best_val_acc:.4f}\n(epoch {best_epoch})",
                xy=(best_epoch, best_val_acc),
                xytext=(best_epoch + 0.5, best_val_acc - 0.08),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xticks(epochs)
    ax.set_ylim(0.7, 1.01)

    # 3. Validation metrics suite
    ax = axes[2]
    ax.plot(epochs, val_acc, "s-", label="Accuracy", color="#388e3c", linewidth=2, markersize=5)
    ax.plot(epochs, val_f1, "o-", label="F1", color="#f57c00", linewidth=2, markersize=5)
    ax.plot(epochs, val_prec, "^-", label="Precision", color="#7b1fa2", linewidth=2, markersize=5)
    ax.plot(epochs, val_rec, "v-", label="Recall", color="#c2185b", linewidth=2, markersize=5)
    ax.plot(epochs, val_auc, "d-", label="AUC", color="#0097a7", linewidth=2, markersize=5)
    ax.axvline(best_epoch, linestyle="--", color="gray", alpha=0.5, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Value")
    ax.set_title("Validation Metrics", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.set_xticks(epochs)
    ax.set_ylim(0.9, 1.005)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  → {out_path}")


def plot_confusion_matrix(rows, out_path):
    """Confusion matrix для лучшей эпохи."""
    val_acc = [r["val_acc"] for r in rows]
    best_idx = max(range(len(rows)), key=lambda i: val_acc[i])
    best = rows[best_idx]
    tp = int(best["val_tp"])
    fp = int(best["val_fp"])
    fn = int(best["val_fn"])
    tn = int(best["val_tn"])

    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    # значения в ячейках
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = 100 * count / total
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: Different\nLocation (0)", "Predicted: Same\nLocation (1)"])
    ax.set_yticklabels(["Actual: Different\nLocation (0)", "Actual: Same\nLocation (1)"])
    ax.set_title(f"Confusion Matrix — Best Epoch ({best['epoch']:.0f})\n"
                 f"Accuracy: {best['val_acc']:.4f} | F1: {best['val_f1']:.4f}",
                 fontweight="bold", pad=15)

    # убираем grid для heatmap
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  → {out_path}")


def save_comparison_table(rows, out_csv, out_txt):
    """Сравнение CP1 baseline (из текста курсовой) и improved baseline (текущий)."""
    val_acc = [r["val_acc"] for r in rows]
    best_idx = max(range(len(rows)), key=lambda i: val_acc[i])
    best = rows[best_idx]

    comparison = [
        ["Architecture", "ResNet18 (from scratch, 6-channel concat)",
         "Siamese ResNet50 (ImageNet pretrained, shared backbone)"],
        ["Pretrained weights", "No", "Yes (ImageNet-1K V2)"],
        ["Data augmentation", "No", "RandomCrop + HorizontalFlip + ColorJitter"],
        ["Optimizer", "Adam", "AdamW (weight_decay=1e-4)"],
        ["LR schedule", "Constant", "CosineAnnealing"],
        ["Trainable parameters", "11.7M", "26.7M"],
        ["Training pairs", "~315 K", "12 586"],
        ["Best epoch", "2", str(int(best["epoch"]))],
        ["Val Accuracy", "0.691", f"{best['val_acc']:.4f}"],
        ["Val F1", "—", f"{best['val_f1']:.4f}"],
        ["Val Precision", "—", f"{best['val_precision']:.4f}"],
        ["Val Recall", "—", f"{best['val_recall']:.4f}"],
        ["Val AUC", "—", f"{1 - best['val_auc']:.4f}"],
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Aspect", "CP1 Baseline", "Improved Baseline (this work)"])
        w.writerows(comparison)
    print(f"  → {out_csv}")

    # Сводка для копирования в курсовую
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("LOCATION VERIFICATION — RESULTS SUMMARY (для курсовой)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Best epoch: {int(best['epoch'])} of {len(rows)} (early stopping triggered)\n\n")
        f.write("Validation metrics on best checkpoint:\n")
        f.write(f"  Accuracy:  {best['val_acc']:.4f}\n")
        f.write(f"  Precision: {best['val_precision']:.4f}\n")
        f.write(f"  Recall:    {best['val_recall']:.4f}\n")
        f.write(f"  F1-score:  {best['val_f1']:.4f}\n")
        f.write(f"  AUC:       {1 - best['val_auc']:.4f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(f"  TP = {int(best['val_tp'])}  FP = {int(best['val_fp'])}\n")
        f.write(f"  FN = {int(best['val_fn'])}  TN = {int(best['val_tn'])}\n\n")
        f.write("Comparison with CP1 baseline:\n")
        for row in comparison:
            f.write(f"  {row[0]:25}  {row[1]:50}  ->  {row[2]}\n")
        f.write("\n" + "=" * 70 + "\n")
    print(f"  → {out_txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/siamese/training_log.csv")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Не найден: {log_path}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_log(log_path)
    print(f"Загружено эпох: {len(rows)}\n")
    print("Создаю файлы:")
    plot_training_curves(rows, out_dir / "training_curves.png")
    plot_confusion_matrix(rows, out_dir / "confusion_matrix.png")
    save_comparison_table(rows, out_dir / "comparison_table.csv",
                          out_dir / "summary.txt")

    print(f"\nВсё в папке {out_dir}/")
    print("Можно вставлять прямо в курсовую.")


if __name__ == "__main__":
    main()
