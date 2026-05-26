#!/usr/bin/env python3
"""
Детектор аномалий в фото «после» через CLIP zero-shot.

Отлавливает:
  - real_photo       — нормальное фото улицы / тротуара / газона
  - ui_screenshot    — скриншот ГИС, карты, базы данных, любого UI
  - document_photo   — сфотографированный документ, акт, лист А4
  - signage_photo    — фото знака или таблички крупным планом (тоже не годится)

Для каждого фото CLIP считает similarity к набору англоязычных промптов
из каждой категории. Финальная категория = с максимальным средним score.

Запуск:
    python3 detect_anomalies.py --csv pair_decisions.csv

Создаёт:
    anomaly_scores.csv  — на каждую пару: predicted_category + scores
    suspicious.csv      — отсортированный список подозрительных для ручной проверки

Зависимости (один раз):
    pip install transformers torch torchvision Pillow
"""

import argparse
import csv
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("Нужны библиотеки: pip install transformers torch torchvision")
    sys.exit(1)


# Промпты подобраны под датасет: Москва, нарушения благоустройства
# (асфальт, газон, бортовой камень) + типичные «не-фото» которые встречаются
CATEGORIES = {
    "real_photo": [
        "an outdoor photograph of an asphalt road or sidewalk",
        "a photo of urban pavement with potholes or damage",
        "a photograph of a city street, curb or lawn",
        "an outdoor street view with buildings and infrastructure",
        "a photo of grass, soil or vegetation in a city",
        "a real-world photograph taken outside on a phone camera",
    ],
    "ui_screenshot": [
        "a screenshot of a web-based map with colored regions",
        "a screenshot of a geographic information system interface",
        "a digital cartographic map with information overlay panels",
        "a screenshot of a computer database or administrative system",
        "a screenshot of a software application window with buttons and menus",
    ],
    "document_photo": [
        "a photograph of a printed paper document or form",
        "a photo of an A4 page with text and signatures",
        "a scanned page of an official report or invoice",
        "a photo of a stack of paperwork on a desk",
    ],
    "signage_photo": [
        "a close-up photo of a road sign or street signage",
        "a photograph of an information board or notice",
    ],
}


def url_to_cache_path(url):
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        ext = ".jpg"
    return Path("image_cache") / f"{h}{ext}"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="pair_decisions.csv")
    ap.add_argument("--out-csv", default="anomaly_scores.csv")
    ap.add_argument("--suspicious-csv", default="suspicious.csv")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--model", default="openai/clip-vit-base-patch32",
                    help="openai/clip-vit-base-patch32 (быстро) или openai/clip-vit-large-patch14 (точнее)")
    ap.add_argument("--only-good", action="store_true",
                    help="проверять только пары с decision='good'")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Загружаю CLIP ({args.model})...")
    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.eval()

    # Подготовка text embeddings (один раз)
    all_prompts = []
    prompt_to_cat = []
    for cat, prompts in CATEGORIES.items():
        for p in prompts:
            all_prompts.append(p)
            prompt_to_cat.append(cat)

    with torch.no_grad():
        text_inputs = processor(text=all_prompts, return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Маски по категориям для усреднения
    cat_masks = {}
    for cat in CATEGORIES:
        cat_masks[cat] = torch.tensor([c == cat for c in prompt_to_cat], device=device)

    # Прочитать пары
    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.only_good:
        rows = [r for r in rows if r.get("decision") == "good"]
    print(f"Пар для проверки: {len(rows)}")

    results = []
    t0 = time.time()

    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        images = []
        valid_rows = []
        for r in batch:
            path = url_to_cache_path(r["after_url"])
            if not path.exists():
                continue
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_rows.append(r)
            except Exception:
                continue

        if not images:
            continue

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt").to(device)
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T) * 100  # [n_img, n_prompts]

            for k, r in enumerate(valid_rows):
                scores = {}
                for cat, mask in cat_masks.items():
                    scores[cat] = similarity[k][mask].mean().item()
                predicted = max(scores, key=scores.get)
                results.append({
                    "source": r["source"],
                    "row": r["row"],
                    "task": r.get("task", ""),
                    "after_url": r["after_url"],
                    "predicted_category": predicted,
                    **{f"score_{c}": round(scores[c], 2) for c in CATEGORIES},
                })

        done = i + len(batch)
        if done % 100 < args.batch_size or done >= len(rows):
            elapsed = time.time() - t0
            rate = done / max(0.001, elapsed)
            eta = (len(rows) - done) / max(0.001, rate)
            print(f"  {done}/{len(rows)}  {rate:.1f} img/sec  ETA {eta:.0f}s",
                  end="\r", flush=True)
    print()

    # Сохранить полную таблицу
    fields = list(results[0].keys())
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nСохранено: {args.out_csv}")

    # Статистика
    cnt = Counter(r["predicted_category"] for r in results)
    print("\nИтог предсказаний:")
    for cat, n in cnt.most_common():
        print(f"  {cat:18}: {n:5}  ({100*n/len(results):5.1f}%)")

    # Подозрительные = всё что не real_photo, отсортировано по score_real_photo возрастающе
    suspicious = [r for r in results if r["predicted_category"] != "real_photo"]
    suspicious.sort(key=lambda r: r["score_real_photo"])
    with open(args.suspicious_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(suspicious)
    print(f"\nПодозрительных (не real_photo): {len(suspicious)}")
    print(f"Сохранены в {args.suspicious_csv} (отсортированы — самые подозрительные сверху)")

    # Топ-15 на экран
    print("\nТоп-15 самых подозрительных:")
    print(f"  {'row':>6}  {'predicted':18}  {'real':>5}  {'ui':>5}  {'doc':>5}  {'sign':>5}  task")
    for r in suspicious[:15]:
        print(f"  {r['row']:>6}  {r['predicted_category']:18}  "
              f"{r['score_real_photo']:5.1f}  {r['score_ui_screenshot']:5.1f}  "
              f"{r['score_document_photo']:5.1f}  {r['score_signage_photo']:5.1f}  "
              f"{r['task'][:40]}")


if __name__ == "__main__":
    main()
