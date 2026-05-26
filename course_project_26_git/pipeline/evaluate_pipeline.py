#!/usr/bin/env python3
"""
Оценка пайплайна на размеченных данных С УЧЁТОМ DATA LEAKAGE.

Разделяет pair_decisions.csv на:
  - SEEN:   пары которые были в train_*.csv или val_*.csv (модель видела)
  - UNSEEN: пары которых модели НЕ видели (истинная производительность)

Считает метрики отдельно для каждой группы. Главное для курсовой = UNSEEN.

Запуск:
    python evaluate_pipeline.py --csv pair_decisions.csv

Автоматически ищет в текущей папке файлы:
    train_asphalt.csv, val_asphalt.csv, train_lawn.csv, val_lawn.csv,
    train_curb.csv, val_curb.csv, train_qa_*.csv, val_qa_*.csv
    train.csv, val.csv (если есть от unified)
и считает их как "уже использованные для обучения".
"""

import argparse
import csv
import hashlib
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
CACHE_DIR = Path("image_cache")

CATEGORY_KEYWORDS = {
    'asphalt': ['асфальт', 'выбоин', 'ям', 'покрытии'],
    'lawn':    ['газон', 'убранный'],
    'curb':    ['бортов', 'камн'],
}

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


def detect_category(task_name):
    t = str(task_name).lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in t for k in keywords):
            return cat
    return 'unknown'


def url_to_cache_path(url):
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        ext = ".jpg"
    return CACHE_DIR / f"{h}{ext}"


def load_model(path, device):
    m = SiameseModel().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def decide(loc_prob, qa_prob, accept_loc=0.95, accept_qa=0.8, reject_loc=0.3, reject_qa=0.3):
    if loc_prob < reject_loc:
        return 'REJECT'
    if qa_prob < reject_qa:
        return 'REJECT'
    if loc_prob > accept_loc and qa_prob > accept_qa:
        return 'ACCEPT'
    return 'HUMAN_REVIEW'


def collect_seen_paths(training_csv_files):
    """Собирает set путей фото которые встречаются в training/val CSV файлах."""
    seen = set()
    for csv_path in training_csv_files:
        if not Path(csv_path).exists():
            continue
        with open(csv_path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                seen.add(r.get('path_before', ''))
                seen.add(r.get('path_after', ''))
    return seen


def is_pair_seen(before_url, after_url, seen_paths):
    """Пара 'seen' если хоть одно из её фото встречалось в training."""
    bp = str(url_to_cache_path(before_url).absolute()).replace('\\', '/').lower()
    ap = str(url_to_cache_path(after_url).absolute()).replace('\\', '/').lower()
    # normalize seen paths the same way
    seen_norm = {str(Path(p).absolute()).replace('\\', '/').lower() for p in seen_paths if p}
    return bp in seen_norm or ap in seen_norm


def run_pipeline_on_pair(r, loc_models, qa_models, mode, device, transform):
    b_url = r['before_url']
    a_url = r['after_url']
    task = r.get('task', '')
    cat = detect_category(task)
    gt = r['decision']

    result = {
        'source': r.get('source', ''), 'row': r.get('row', ''),
        'task': task, 'category': cat, 'ground_truth': gt,
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

    b_path = url_to_cache_path(b_url)
    a_path = url_to_cache_path(a_url)
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
    valid = [r for r in results if r['decision'] in ('ACCEPT', 'REJECT', 'HUMAN_REVIEW')]
    if not valid:
        return None

    cm = {('good', 'ACCEPT'): 0, ('good', 'REJECT'): 0, ('good', 'HUMAN_REVIEW'): 0,
          ('bad',  'ACCEPT'): 0, ('bad',  'REJECT'): 0, ('bad',  'HUMAN_REVIEW'): 0}
    for r in valid:
        gt = r['ground_truth']
        if gt in ('good', 'bad'):
            cm[(gt, r['decision'])] = cm.get((gt, r['decision']), 0) + 1

    good_total = sum(v for (gt, _), v in cm.items() if gt == 'good')
    bad_total = sum(v for (gt, _), v in cm.items() if gt == 'bad')
    total = good_total + bad_total

    correct_accept = cm[('good', 'ACCEPT')]
    correct_reject = cm[('bad', 'REJECT')]
    sent_to_human = cm[('good', 'HUMAN_REVIEW')] + cm[('bad', 'HUMAN_REVIEW')]
    wrong_accept = cm[('bad', 'ACCEPT')]
    wrong_reject = cm[('good', 'REJECT')]

    auto_decisions = correct_accept + correct_reject + wrong_accept + wrong_reject
    auto_rate = auto_decisions / max(1, total)
    auto_acc = (correct_accept + correct_reject) / max(1, auto_decisions)
    fp_rate = wrong_accept / max(1, bad_total)
    human_rate = sent_to_human / max(1, total)

    return {
        'n_total': total, 'good_total': good_total, 'bad_total': bad_total,
        'confusion': cm,
        'correct_accept': correct_accept, 'correct_reject': correct_reject,
        'wrong_accept': wrong_accept, 'wrong_reject': wrong_reject,
        'sent_to_human': sent_to_human,
        'auto_rate': auto_rate, 'auto_acc': auto_acc,
        'fp_rate': fp_rate, 'human_rate': human_rate,
    }


def print_metrics(metrics, label):
    if metrics is None:
        print(f"\n  [{label}] — no data"); return
    print(f"\n  --- [{label}] ---")
    print(f"  pairs: {metrics['n_total']} (good={metrics['good_total']}, bad={metrics['bad_total']})")
    print(f"  decisions: ACCEPT={metrics['correct_accept']+metrics['wrong_accept']}  "
          f"REJECT={metrics['correct_reject']+metrics['wrong_reject']}  "
          f"HUMAN={metrics['sent_to_human']}")
    print(f"  confusion:")
    print(f"    GT=good  → ACCEPT:{metrics['confusion'][('good','ACCEPT')]:>4}  "
          f"REJECT:{metrics['confusion'][('good','REJECT')]:>4}  "
          f"HUMAN:{metrics['confusion'][('good','HUMAN_REVIEW')]:>4}")
    print(f"    GT=bad   → ACCEPT:{metrics['confusion'][('bad','ACCEPT')]:>4}  "
          f"REJECT:{metrics['confusion'][('bad','REJECT')]:>4}  "
          f"HUMAN:{metrics['confusion'][('bad','HUMAN_REVIEW')]:>4}")
    print(f"  KEY METRICS:")
    print(f"    auto-decision rate: {100*metrics['auto_rate']:5.1f}%  (% решено автоматически)")
    print(f"    auto accuracy:      {100*metrics['auto_acc']:5.1f}%  (когда решаем сами — правильно)")
    print(f"    human review rate:  {100*metrics['human_rate']:5.1f}%  (% к человеку)")
    print(f"    critical FP rate:   {100*metrics['fp_rate']:5.1f}%  (bad пары, прошедшие как ACCEPT)")


def load_models_for_mode(mode, device):
    loc_models, qa_models = {}, {}
    if mode == 'unified':
        if Path(UNIFIED_LOCATION_PATH).exists():
            loc_models['unified'] = load_model(UNIFIED_LOCATION_PATH, device)
    else:
        for cat in ['asphalt', 'lawn', 'curb']:
            p = MODELS_PATHS[cat]['location']
            if Path(p).exists():
                loc_models[cat] = load_model(p, device)
    for cat in ['asphalt', 'lawn', 'curb']:
        p = MODELS_PATHS[cat]['quality']
        if Path(p).exists():
            qa_models[cat] = load_model(p, device)
    return loc_models, qa_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="pair_decisions.csv")
    ap.add_argument("--modes", nargs='+', default=['specialized', 'unified'])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-summary", default="evaluation_summary.txt")
    ap.add_argument("--out-csv", default="evaluation_results.csv")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # === Шаг 1: загрузить ground truth ===
    with open(args.csv, encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['decision'] in ('good', 'bad')]
    if args.limit:
        rows = rows[:args.limit]
    print(f"\nPairs with ground truth: {len(rows)}")

    # === Шаг 2: определить SEEN vs UNSEEN ===
    print("\n=== Поиск пар которые были в обучающих CSV (data leakage check) ===")
    training_csvs = [
        'train_asphalt.csv', 'val_asphalt.csv',
        'train_lawn.csv',    'val_lawn.csv',
        'train_curb.csv',    'val_curb.csv',
        'train_qa_asphalt.csv', 'val_qa_asphalt.csv',
        'train_qa_lawn.csv',    'val_qa_lawn.csv',
        'train_qa_curb.csv',    'val_qa_curb.csv',
        'train.csv', 'val.csv',  # unified
    ]
    found_csvs = [c for c in training_csvs if Path(c).exists()]
    print(f"  Найдены обучающие CSV: {len(found_csvs)}")
    for c in found_csvs:
        print(f"    - {c}")

    seen_paths = collect_seen_paths(found_csvs)
    print(f"  Уникальных фото в обучении: {len(seen_paths)}")

    for r in rows:
        r['_is_seen'] = is_pair_seen(r['before_url'], r['after_url'], seen_paths)

    seen_rows = [r for r in rows if r['_is_seen']]
    unseen_rows = [r for r in rows if not r['_is_seen']]
    print(f"\n  SEEN pairs:   {len(seen_rows):>5}  (модель видела хотя бы одно фото)")
    print(f"  UNSEEN pairs: {len(unseen_rows):>5}  (модель НЕ видела ни одного фото) ← основное для курсовой")

    if len(unseen_rows) < 50:
        print(f"\n  ⚠️  ВНИМАНИЕ: всего {len(unseen_rows)} unseen пар — мало для статистики.")
        print(f"     Метрики на этой выборке могут быть шумными.")

    # === Шаг 3: для каждого режима — прогон ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    all_results = {}
    all_metrics = {}

    for mode in args.modes:
        print(f"\n{'='*70}")
        print(f"  MODE: {mode}")
        print(f"{'='*70}")
        loc_models, qa_models = load_models_for_mode(mode, device)
        if not loc_models or not qa_models:
            print(f"  Skipping {mode}: missing models")
            continue
        print(f"  Loaded: {len(loc_models)} loc models, {len(qa_models)} qa models")

        print(f"  Running pipeline...")
        t0 = time.time()
        results = []
        for i, r in enumerate(rows):
            res = run_pipeline_on_pair(r, loc_models, qa_models, mode, device, transform)
            res['is_seen'] = r['_is_seen']
            results.append(res)
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(rows)}  ({(i+1)/(time.time()-t0):.1f}/sec)", end='\r', flush=True)
        print(f"\n  Done in {int(time.time()-t0)}s")

        seen_results = [r for r in results if r.get('is_seen')]
        unseen_results = [r for r in results if not r.get('is_seen')]

        seen_metrics = compute_metrics(seen_results)
        unseen_metrics = compute_metrics(unseen_results)
        overall_metrics = compute_metrics(results)

        print_metrics(unseen_metrics, "UNSEEN — главное для курсовой")
        print_metrics(seen_metrics, "SEEN — sanity check, должно быть выше unseen")
        print_metrics(overall_metrics, "ALL — общая")

        all_results[mode] = results
        all_metrics[mode] = {
            'seen': seen_metrics, 'unseen': unseen_metrics, 'overall': overall_metrics
        }

    # === Шаг 4: сохранить ===
    if all_results:
        merged = {}
        for mode, results in all_results.items():
            for r in results:
                key = (r['source'], r['row'])
                if key not in merged:
                    merged[key] = {
                        'source': r['source'], 'row': r['row'],
                        'task': r['task'], 'category': r['category'],
                        'ground_truth': r['ground_truth'],
                        'is_seen': r.get('is_seen', False),
                    }
                merged[key][f'{mode}_loc_prob'] = r.get('location_prob')
                merged[key][f'{mode}_qa_prob'] = r.get('quality_prob')
                merged[key][f'{mode}_decision'] = r.get('decision')
        pd.DataFrame(list(merged.values())).to_csv(args.out_csv, index=False, encoding='utf-8')
        print(f"\nPer-pair: {args.out_csv}")

    with open(args.out_summary, 'w', encoding='utf-8') as f:
        f.write("PIPELINE EVALUATION — SEEN vs UNSEEN (data leakage adjusted)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Found {len(seen_rows)} SEEN pairs (was in training), "
                f"{len(unseen_rows)} UNSEEN pairs (true test set)\n\n")

        for mode, metrics_dict in all_metrics.items():
            f.write(f"\n--- Mode: {mode} ---\n")
            for split in ['unseen', 'seen', 'overall']:
                m = metrics_dict.get(split)
                if m is None: continue
                f.write(f"\n  [{split.upper()}] n={m['n_total']} (good={m['good_total']}, bad={m['bad_total']})\n")
                f.write(f"    auto_accuracy:     {100*m['auto_acc']:6.2f}%\n")
                f.write(f"    auto_decision_rate:{100*m['auto_rate']:6.2f}%\n")
                f.write(f"    human_review_rate: {100*m['human_rate']:6.2f}%\n")
                f.write(f"    critical_FP_rate:  {100*m['fp_rate']:6.2f}%\n")
                f.write(f"    correct_accept: {m['correct_accept']}, correct_reject: {m['correct_reject']}\n")
                f.write(f"    wrong_accept:   {m['wrong_accept']} (CRITICAL), wrong_reject: {m['wrong_reject']}\n")

        if 'specialized' in all_metrics and 'unified' in all_metrics:
            f.write("\n\n--- COMPARISON specialized vs unified (UNSEEN only) ---\n")
            s = all_metrics['specialized']['unseen']
            u = all_metrics['unified']['unseen']
            if s and u:
                f.write(f"  auto_accuracy:    specialized={100*s['auto_acc']:.2f}%  "
                        f"unified={100*u['auto_acc']:.2f}%\n")
                f.write(f"  critical_FP_rate: specialized={100*s['fp_rate']:.2f}%  "
                        f"unified={100*u['fp_rate']:.2f}%\n")
                f.write(f"  human_review:     specialized={100*s['human_rate']:.2f}%  "
                        f"unified={100*u['human_rate']:.2f}%\n")
                better = 'specialized' if s['auto_acc'] > u['auto_acc'] else 'unified'
                f.write(f"\n  WINNER (по auto_accuracy на UNSEEN): {better}\n")
    print(f"Summary: {args.out_summary}")


if __name__ == "__main__":
    main()
