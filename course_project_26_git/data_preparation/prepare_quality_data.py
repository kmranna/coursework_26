#!/usr/bin/env python3
"""
Подготовка данных для Quality Assessment модели — из уже готовых CSV.

Берёт positive pairs (label=1, то есть реальные пары одного места) из train.csv
и val.csv (сформированных prepare_training_data.py), и создаёт для каждой
quality pair:
  - Positive (label=1, "исправлено"): тот же (before, after) что и в исходном CSV
  - Negative (label=0, "не исправлено", synthetic=1): (before, before), к которому
    train_quality.py применит специальную аугментацию имитирующую "то же место,
    переснято, но НЕ исправлено"

Запуск:
    python prepare_quality_data.py --train-csv train_asphalt.csv \
                                   --val-csv val_asphalt.csv \
                                   --out-train train_qa_asphalt.csv \
                                   --out-val val_qa_asphalt.csv

Никаких загрузок — только переформирует CSV. Работает за секунды.
"""

import argparse
import csv
import random
from pathlib import Path


def load_positives(csv_path):
    rows = []
    with open(csv_path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['label'] == '1':
                rows.append(r)
    return rows


def make_quality_pairs(positives, category='unknown'):
    out = []
    for r in positives:
        before = r['path_before']
        after = r['path_after']
        task = r.get('task', '')
        cat = r.get('category', category)

        out.append({
            'path_before': before,
            'path_after': after,
            'label': 1,
            'synthetic': 0,
            'category': cat,
            'task': task,
        })
        out.append({
            'path_before': before,
            'path_after': before,
            'label': 0,
            'synthetic': 1,
            'category': cat,
            'task': task,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--out-train", default="train_qa.csv")
    ap.add_argument("--out-val", default="val_qa.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=== Чтение исходных CSV ===")
    train_pos = load_positives(args.train_csv)
    val_pos = load_positives(args.val_csv)
    print(f"  {args.train_csv}: positive pairs = {len(train_pos)}")
    print(f"  {args.val_csv}:   positive pairs = {len(val_pos)}")

    if not train_pos or not val_pos:
        print("ОШИБКА: в CSV нет positive pairs (label=1). Проверь файлы.")
        return

    category = train_pos[0].get('category', Path(args.train_csv).stem.replace('train_', ''))
    print(f"  Категория: {category}")

    print("\n=== Формирование Quality pairs ===")
    train_qa = make_quality_pairs(train_pos, category=category)
    val_qa = make_quality_pairs(val_pos, category=category)

    random.seed(args.seed)
    random.shuffle(train_qa)
    random.shuffle(val_qa)

    tp = sum(1 for r in train_qa if r['label'] == 1)
    tn = sum(1 for r in train_qa if r['label'] == 0)
    vp = sum(1 for r in val_qa if r['label'] == 1)
    vn = sum(1 for r in val_qa if r['label'] == 0)
    print(f"  Train QA: {len(train_qa)} pairs (pos {tp}, synthetic neg {tn})")
    print(f"  Val QA:   {len(val_qa)} pairs (pos {vp}, synthetic neg {vn})")

    fields = ['path_before', 'path_after', 'label', 'synthetic', 'category', 'task']
    with open(args.out_train, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(train_qa)
    with open(args.out_val, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(val_qa)

    print("\nГотово!")
    print(f"  Создано: {args.out_train}, {args.out_val}")
    print("\nДальше — обучение:")
    print(f"  python train_quality.py --train-csv {args.out_train} --val-csv {args.out_val} \\")
    print(f"      --epochs 10 --batch-size 32 --workers 0 --out-dir runs\\qa_{category}")


if __name__ == "__main__":
    main()
