#!/usr/bin/env python3
"""
Оценка end-to-end пайплайна на val_*.csv файлах (location verification).

Использует уже существующие пути к фото (не качает ничего).
Ground truth интерпретация:
  label=1 (real positive pair) → ожидаем ACCEPT (это реальная accepted задача)
  label=0 (shuffled negative)  → ожидаем REJECT (это разные места)

Сравнивает specialized vs unified Location режимы. Quality model применяется
к обоим (Quality всегда specialized по категории).

Запуск:
    python evaluate_on_val.py
    python evaluate_on_val.py --modes specialized unified

Скрипт автоматически ищет в текущей папке val_asphalt.csv, val_lawn.csv, val_curb.csv.
"""

import argparse
import csv
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.models as models

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODELS_PATHS = {
    'asphalt': {'location': 'runs/asphalt/best_model.pt',
                'quality':  'runs/qa_asphalt/best_model.pt'},
    'lawn':    {'location': 'runs/lawn/best_model.pt',
                'quality':  'runs/qa_lawn/best_model.pt'},
    'curb':    {'location': 'runs/curb/best_model.pt',
                'quality':  'runs/qa_curb/best_model.pt'},
}
UNIFIED_LOCATION_PATH = 'runs/siamese_full/best_model.pt'


class SiameseModel(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        net = models.resnet50(weights=None)
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
        return self.head(torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1)).squeeze(1)


def load_model(path, device):
    m = SiameseModel().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # best_model.pt может быть либо плоским state_dict, либо checkpoint dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    m.load_state_dict(state)
    m.eval()
    return m


def load_models_for_mode(mode, device):
    loc_models, qa_models = {}, {}
    if mode == 'unified':
        if Path(UNIFIED_LOCATION_PATH).exists():
            loc_models['unified'] = load_model(UNIFIED_LOCATION_PATH, device)
            print(f"  ✓ Loaded unified Location")
    else:
        for cat in ['asphalt', 'lawn', 'curb']:
            p = MODELS_PATHS[cat]['location']
            if Path(p).exists():
                loc_models[cat] = load_model(p, device)
                print(f"  ✓ Loaded Location {cat}")
    for cat in ['asphalt', 'lawn', 'curb']:
        p = MODELS_PATHS[cat]['quality']
        if Path(p).exists():
            qa_models[cat] = load_model(p, device)
            print(f"  ✓ Loaded Quality {cat}")
    return loc_models, qa_models


def decide(loc_prob, qa_prob, accept_loc=0.95, accept_qa=0.8, reject_loc=0.3, reject_qa=0.3):
    if loc_prob < reject_loc:
        return 'REJECT'
    if qa_prob < reject_qa:
        return 'REJECT'
    if loc_prob > accept_loc and qa_prob > accept_qa:
        return 'ACCEPT'
    return 'HUMAN_REVIEW'


def expected_decision(label):
    """label=1 = real positive pair → ACCEPT; label=0 = shuffled negative → REJECT."""
    return 'ACCEPT' if int(label) == 1 else 'REJECT'


def run_on_row(r, loc_models, qa_models, mode, device, transform):
    cat = r.get('category', 'unknown')
    b_path = Path(r['path_before'])
    a_path = Path(r['path_after'])
    label = int(r['label'])
    expected = expected_decision(label)

    result = {
        'category': cat, 'label': label, 'expected': expected,
        'location_prob': None, 'quality_prob': None, 'decision': 'ERROR',
    }

    if mode == 'unified':
        loc_model = loc_models.get('unified')
    else:
        loc_model = loc_models.get(cat)
    qa_model = qa_models.get(cat)

    if loc_model is None or qa_model is None:
        result['decision'] = 'NO_MODEL'
        return result
    if not (b_path.exists() and a_path.exists()):
        result['decision'] = 'NO_PHOTO'
        return result

    try:
        img_b = transform(Image.open(b_path).convert('RGB')).unsqueeze(0).to(device)
        img_a = transform(Image.open(a_path).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            loc_prob = torch.sigmoid(loc_model(img_b, img_a)).item()
            qa_prob = torch.sigmoid(qa_model(img_b, img_a)).item()
        result['location_prob'] = round(loc_prob, 4)
        result['quality_prob'] = round(qa_prob, 4)
        result['decision'] = decide(loc_prob, qa_prob)
    except Exception:
        result['decision'] = 'ERROR'
    return result


def compute_metrics(results):
    """label=1 → expected ACCEPT, label=0 → expected REJECT."""
    valid = [r for r in results if r['decision'] in ('ACCEPT', 'REJECT', 'HUMAN_REVIEW')]
    if not valid:
        return None

    cm = {(1, 'ACCEPT'): 0, (1, 'REJECT'): 0, (1, 'HUMAN_REVIEW'): 0,
          (0, 'ACCEPT'): 0, (0, 'REJECT'): 0, (0, 'HUMAN_REVIEW'): 0}
    for r in valid:
        cm[(r['label'], r['decision'])] = cm.get((r['label'], r['decision']), 0) + 1

    pos_total = sum(v for (l, _), v in cm.items() if l == 1)
    neg_total = sum(v for (l, _), v in cm.items() if l == 0)
    total = pos_total + neg_total

    correct_accept = cm[(1, 'ACCEPT')]    # положительная → правильно ACCEPT
    correct_reject = cm[(0, 'REJECT')]    # отрицательная → правильно REJECT
    wrong_accept = cm[(0, 'ACCEPT')]      # отрицательная → ошибочно ACCEPT (КРИТИЧНО)
    wrong_reject = cm[(1, 'REJECT')]      # положительная → ошибочно REJECT
    human = cm[(1, 'HUMAN_REVIEW')] + cm[(0, 'HUMAN_REVIEW')]

    auto = correct_accept + correct_reject + wrong_accept + wrong_reject
    auto_rate = auto / max(1, total)
    auto_acc = (correct_accept + correct_reject) / max(1, auto)
    fp_rate = wrong_accept / max(1, neg_total)
    human_rate = human / max(1, total)

    return {
        'n_total': total, 'pos_total': pos_total, 'neg_total': neg_total,
        'confusion': cm,
        'correct_accept': correct_accept, 'correct_reject': correct_reject,
        'wrong_accept': wrong_accept, 'wrong_reject': wrong_reject,
        'sent_to_human': human,
        'auto_rate': auto_rate, 'auto_acc': auto_acc,
        'fp_rate': fp_rate, 'human_rate': human_rate,
    }


def print_metrics(metrics, label):
    if metrics is None:
        print(f"\n  [{label}] no data"); return
    print(f"\n  --- [{label}] ---")
    print(f"  pairs: {metrics['n_total']} (positive label=1: {metrics['pos_total']}, "
          f"negative label=0: {metrics['neg_total']})")
    print(f"  confusion:")
    print(f"    label=1 (real pair, expected ACCEPT)  → ACCEPT:{metrics['confusion'][(1,'ACCEPT')]:>5}  "
          f"REJECT:{metrics['confusion'][(1,'REJECT')]:>5}  HUMAN:{metrics['confusion'][(1,'HUMAN_REVIEW')]:>5}")
    print(f"    label=0 (shuffled,  expected REJECT)  → ACCEPT:{metrics['confusion'][(0,'ACCEPT')]:>5}  "
          f"REJECT:{metrics['confusion'][(0,'REJECT')]:>5}  HUMAN:{metrics['confusion'][(0,'HUMAN_REVIEW')]:>5}")
    print(f"  KEY METRICS:")
    print(f"    auto-decision rate: {100*metrics['auto_rate']:5.1f}%  (% решено автоматически)")
    print(f"    auto accuracy:      {100*metrics['auto_acc']:5.1f}%  (% правильных автоматических решений)")
    print(f"    human review rate:  {100*metrics['human_rate']:5.1f}%  (% к человеку)")
    print(f"    critical FP rate:   {100*metrics['fp_rate']:5.1f}%  (% shuffled пар прошедших как ACCEPT)")


def load_val_csvs(paths):
    rows = []
    for p in paths:
        if not Path(p).exists():
            print(f"  ✗ not found: {p}")
            continue
        # Определяем категорию из имени файла: val_asphalt.csv → asphalt
        name = Path(p).stem.lower()
        category = None
        for cat in ['asphalt', 'lawn', 'curb']:
            if cat in name:
                category = cat
                break
        n_before = len(rows)
        with open(p, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                # Если в CSV нет/пустая категория — берём из имени файла
                if not r.get('category') or r.get('category') in ('unknown', ''):
                    if category:
                        r['category'] = category
                rows.append(r)
        print(f"  ✓ {p}  (+{len(rows) - n_before} rows, category={category})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csvs", nargs='+',
                    default=['val_asphalt.csv', 'val_lawn.csv', 'val_curb.csv'])
    ap.add_argument("--modes", nargs='+', default=['specialized', 'unified'])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-summary", default="evaluation_summary.txt")
    ap.add_argument("--out-csv", default="evaluation_results.csv")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print(f"\n=== Загрузка val CSV ===")
    rows = load_val_csvs(args.val_csvs)
    if args.limit:
        rows = rows[:args.limit]
    print(f"\nTotal pairs to evaluate: {len(rows)}")
    if not rows:
        print("Нет данных, выход."); return

    # Распределение
    pos = sum(1 for r in rows if int(r['label']) == 1)
    neg = sum(1 for r in rows if int(r['label']) == 0)
    print(f"  positive (label=1): {pos}")
    print(f"  negative (label=0, shuffled): {neg}")
    cats = {}
    for r in rows:
        c = r.get('category', 'unknown')
        cats[c] = cats.get(c, 0) + 1
    print(f"  by category: {cats}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    all_results = {}
    all_metrics = {}

    for mode in args.modes:
        print(f"\n{'='*70}\n  MODE: {mode}\n{'='*70}")
        loc_models, qa_models = load_models_for_mode(mode, device)
        if not loc_models or not qa_models:
            print("  Missing models, skipping"); continue

        print(f"  Running pipeline...")
        t0 = time.time()
        results = []
        for i, r in enumerate(rows):
            res = run_on_row(r, loc_models, qa_models, mode, device, transform)
            results.append(res)
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(rows)}  ({(i+1)/(time.time()-t0):.1f}/sec)",
                      end='\r', flush=True)
        print(f"\n  Done in {int(time.time()-t0)}s")

        metrics_overall = compute_metrics(results)
        print_metrics(metrics_overall, f"OVERALL — mode={mode}")

        # Per-category breakdown
        for cat in ['asphalt', 'lawn', 'curb']:
            cat_results = [r for r in results if r['category'] == cat]
            if cat_results:
                cm = compute_metrics(cat_results)
                print_metrics(cm, f"{cat.upper()} — mode={mode}")

        all_results[mode] = results
        all_metrics[mode] = metrics_overall

    # Save per-pair results
    if all_results:
        merged = {}
        for mode, results in all_results.items():
            for i, r in enumerate(results):
                key = i
                if key not in merged:
                    merged[key] = {
                        'category': r['category'], 'label': r['label'],
                        'expected': r['expected'],
                    }
                merged[key][f'{mode}_loc_prob'] = r.get('location_prob')
                merged[key][f'{mode}_qa_prob'] = r.get('quality_prob')
                merged[key][f'{mode}_decision'] = r.get('decision')
        pd.DataFrame(list(merged.values())).to_csv(args.out_csv, index=False, encoding='utf-8')
        print(f"\nPer-pair: {args.out_csv}")

    with open(args.out_summary, 'w', encoding='utf-8') as f:
        f.write("PIPELINE EVALUATION — on val_*.csv (held-out from training)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Pairs: {len(rows)} (positive={pos}, shuffled-negative={neg})\n\n")
        for mode, m in all_metrics.items():
            if m is None: continue
            f.write(f"\n--- Mode: {mode} ---\n")
            f.write(f"  auto_accuracy:     {100*m['auto_acc']:6.2f}%\n")
            f.write(f"  auto_decision_rate:{100*m['auto_rate']:6.2f}%\n")
            f.write(f"  human_review_rate: {100*m['human_rate']:6.2f}%\n")
            f.write(f"  critical_FP_rate:  {100*m['fp_rate']:6.2f}%\n")
            f.write(f"  correct_accept: {m['correct_accept']}, correct_reject: {m['correct_reject']}\n")
            f.write(f"  wrong_accept:   {m['wrong_accept']} (CRITICAL), wrong_reject: {m['wrong_reject']}\n")
        if 'specialized' in all_metrics and 'unified' in all_metrics:
            s, u = all_metrics['specialized'], all_metrics['unified']
            if s and u:
                f.write("\n--- COMPARISON specialized vs unified ---\n")
                f.write(f"  auto_accuracy:    specialized={100*s['auto_acc']:.2f}%  unified={100*u['auto_acc']:.2f}%\n")
                f.write(f"  critical_FP_rate: specialized={100*s['fp_rate']:.2f}%  unified={100*u['fp_rate']:.2f}%\n")
                f.write(f"  human_review:     specialized={100*s['human_rate']:.2f}%  unified={100*u['human_rate']:.2f}%\n")
                better = 'specialized' if s['auto_acc'] > u['auto_acc'] else 'unified'
                f.write(f"\n  WINNER: {better}\n")
    print(f"Summary: {args.out_summary}")


if __name__ == "__main__":
    main()
