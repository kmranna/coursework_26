#!/usr/bin/env python3
"""
MVP Pipeline — Excel со ссылками на пары на входе → Excel с решениями на выходе.

Архитектура:
  Input xlsx (с колонками "Название задачи", "Нарушение", "Устранение")
    ↓
  [Category Router] определяет тип нарушения из task name
    ↓
  [Location Verification model] (specialized или unified) → loc_prob
    ↓
  [Quality Assessment model] (specialized по категории) → qa_prob
    ↓
  [Decision Router] применяет пороги:
     loc < 0.3                          → REJECT (не то место)
     qa < 0.3                           → REJECT (не исправлено)
     loc > 0.95 AND qa > 0.8            → ACCEPT
     иначе                              → HUMAN_REVIEW
    ↓
  Output xlsx (добавлены колонки: category, location_prob, quality_prob, decision, reason)

Запуск:
    python mvp_pipeline.py input.xlsx --output decisions.xlsx --mode specialized
    python mvp_pipeline.py input.xlsx --output decisions.xlsx --mode unified
    python mvp_pipeline.py input.xlsx --limit 100  # только первые 100 для теста
"""

import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.models as models

ImageFile.LOAD_TRUNCATED_IMAGES = True

# === Конфигурация ===
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CACHE_DIR = Path("image_cache")

CATEGORY_KEYWORDS = {
    'asphalt': ['асфальт', 'выбоин', 'ям', 'покрытии'],
    'lawn':    ['газон', 'убранный'],
    'curb':    ['бортов', 'камн'],
}

MODELS_PATHS = {
    'asphalt': {
        'location': 'runs/asphalt/best_model.pt',
        'quality':  'runs/qa_asphalt/best_model.pt',
    },
    'lawn': {
        'location': 'runs/lawn/best_model.pt',
        'quality':  'runs/qa_lawn/best_model.pt',
    },
    'curb': {
        'location': 'runs/curb/best_model.pt',
        'quality':  'runs/qa_curb/best_model.pt',
    },
}
UNIFIED_LOCATION_PATH = 'runs/siamese_full/best_model.pt'


# === Модель (та же что в train_siamese.py / train_quality.py) ===
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


# === Helpers ===
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


def download_one(url_path):
    url, path = url_path
    if path.exists():
        return True
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False


def load_model(path, device):
    m = SiameseModel().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def load_all_models(device, mode='specialized', verbose=True):
    loc_models = {}
    qa_models = {}

    if mode == 'unified':
        if Path(UNIFIED_LOCATION_PATH).exists():
            loc_models['unified'] = load_model(UNIFIED_LOCATION_PATH, device)
            if verbose: print(f"  ✓ Unified Location: {UNIFIED_LOCATION_PATH}")
        else:
            if verbose: print(f"  ✗ NOT FOUND: {UNIFIED_LOCATION_PATH}")
    else:
        for cat in ['asphalt', 'lawn', 'curb']:
            p = MODELS_PATHS[cat]['location']
            if Path(p).exists():
                loc_models[cat] = load_model(p, device)
                if verbose: print(f"  ✓ Location {cat}: {p}")
            else:
                if verbose: print(f"  ✗ NOT FOUND: {p}")

    for cat in ['asphalt', 'lawn', 'curb']:
        p = MODELS_PATHS[cat]['quality']
        if Path(p).exists():
            qa_models[cat] = load_model(p, device)
            if verbose: print(f"  ✓ Quality {cat}: {p}")
        else:
            if verbose: print(f"  ✗ NOT FOUND: {p}")

    return loc_models, qa_models


def decide(loc_prob, qa_prob, accept_loc, accept_qa, reject_loc, reject_qa):
    if loc_prob < reject_loc:
        return 'REJECT', f'Different location (loc={loc_prob:.2f})'
    if qa_prob < reject_qa:
        return 'REJECT', f'Quality issue / not fixed (qa={qa_prob:.2f})'
    if loc_prob > accept_loc and qa_prob > accept_qa:
        return 'ACCEPT', f'High confidence (loc={loc_prob:.2f}, qa={qa_prob:.2f})'
    return 'HUMAN_REVIEW', f'Uncertain (loc={loc_prob:.2f}, qa={qa_prob:.2f})'


# === Main ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_xlsx", help="Входной Excel со ссылками")
    ap.add_argument("--output", default="pipeline_output.xlsx")
    ap.add_argument("--mode", choices=['specialized', 'unified'], default='specialized')
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--accept-loc", type=float, default=0.95)
    ap.add_argument("--accept-qa", type=float, default=0.80)
    ap.add_argument("--reject-loc", type=float, default=0.30)
    ap.add_argument("--reject-qa", type=float, default=0.30)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}  |  Mode: {args.mode}")
    print(f"Thresholds: ACCEPT(loc>{args.accept_loc}, qa>{args.accept_qa}), "
          f"REJECT(loc<{args.reject_loc} OR qa<{args.reject_qa})")

    print(f"\n=== Загрузка моделей ===")
    loc_models, qa_models = load_all_models(device, mode=args.mode)

    print(f"\n=== Чтение {args.input_xlsx} ===")
    df = pd.read_excel(args.input_xlsx)
    if args.limit:
        df = df.head(args.limit)
    df = df.reset_index(drop=True)
    print(f"  Задач: {len(df)}")

    df['category'] = df['Название задачи'].apply(detect_category)
    print(f"  Категории: {dict(df['category'].value_counts())}")

    # === Скачивание ===
    CACHE_DIR.mkdir(exist_ok=True)
    download_tasks = []
    for _, row in df.iterrows():
        b_url = str(row['Нарушение'])
        a_url = str(row['Устранение'])
        download_tasks.append((b_url, url_to_cache_path(b_url)))
        download_tasks.append((a_url, url_to_cache_path(a_url)))

    already = sum(1 for _, p in download_tasks if p.exists())
    print(f"\n=== Скачивание фото ({already}/{len(download_tasks)} уже в кэше) ===")
    if already < len(download_tasks):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, _ in enumerate(ex.map(download_one, download_tasks)):
                if (i + 1) % 200 == 0:
                    print(f"  {i+1}/{len(download_tasks)}  elapsed={int(time.time()-t0)}s",
                          end='\r', flush=True)
        print(f"  Готово за {int(time.time()-t0)}s")

    # === Inference ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print(f"\n=== Inference ===")
    results = []
    t0 = time.time()
    for idx, row in df.iterrows():
        b_url = str(row['Нарушение'])
        a_url = str(row['Устранение'])
        cat = row['category']
        task = str(row.get('Название задачи', ''))

        result = {
            'task': task, 'category': cat,
            'before_url': b_url, 'after_url': a_url,
            'location_prob': None, 'quality_prob': None,
            'decision': 'ERROR', 'reason': '',
        }

        # Выбор моделей
        if args.mode == 'unified':
            loc_model = loc_models.get('unified')
        else:
            loc_model = loc_models.get(cat)
        qa_model = qa_models.get(cat)

        if loc_model is None or qa_model is None:
            result['reason'] = f'No model for "{cat}"'
            results.append(result); continue

        b_path = url_to_cache_path(b_url)
        a_path = url_to_cache_path(a_url)
        if not (b_path.exists() and a_path.exists()):
            result['reason'] = 'Photo not available'
            results.append(result); continue

        try:
            img_b = transform(Image.open(b_path).convert('RGB')).unsqueeze(0).to(device)
            img_a = transform(Image.open(a_path).convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                loc_prob = torch.sigmoid(loc_model(img_b, img_a)).item()
                qa_prob = torch.sigmoid(qa_model(img_b, img_a)).item()
            decision, reason = decide(loc_prob, qa_prob,
                                       args.accept_loc, args.accept_qa,
                                       args.reject_loc, args.reject_qa)
            result.update({
                'location_prob': round(loc_prob, 4),
                'quality_prob': round(qa_prob, 4),
                'decision': decision,
                'reason': reason,
            })
        except Exception as e:
            result['reason'] = f'Inference error: {e}'

        results.append(result)
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(df)}  ({(idx+1)/(time.time()-t0):.1f}/sec)", end='\r', flush=True)
    print()

    out_df = pd.DataFrame(results)
    out_df.to_excel(args.output, index=False)

    print(f"\n=== Распределение решений ===")
    print(out_df['decision'].value_counts())
    print(f"\nСохранено: {args.output}")


if __name__ == "__main__":
    main()
