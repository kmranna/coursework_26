#!/usr/bin/env python3
"""
Подготовка train.csv / val.csv для обучения Siamese-модели.

Два режима:

[1] Использовать уже скачанные пары (БЕЗ скачивания):
    python3 prepare_training_data.py --from-decisions pair_decisions.csv

    Берёт все пары с decision='good' из pair_decisions.csv где оба фото
    уже скачаны в image_cache/, генерирует negatives, делает train/val split.

[2] Подготовить из xlsx (со скачиванием):
    python3 prepare_training_data.py file1.xlsx file2.xlsx file3.xlsx --sample 5000

Что общее:
  - Берёт N positive pairs стратифицированно по категории (источнику xlsx).
  - Делает столько же negative pairs внутри той же категории (within-class shuffle).
  - Stratified train/val split по (категория, label).
"""

import argparse
import csv
import hashlib
import random
import sys
import urllib.request
import urllib.error
import socket
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None  # понадобится только в режиме xlsx

CACHE_DIR = Path("image_cache")
TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (prepare_training_data.py)"

# ВАЖНО: ровно та же логика именования, что в filter_pairs.py,
# чтобы пути попадали в уже существующие файлы кэша
KNOWN_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def url_to_cache_path(url: str) -> Path:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in KNOWN_EXTS:
        ext = ".jpg"
    return CACHE_DIR / f"{h}{ext}"


def is_cached(url: str) -> bool:
    p = url_to_cache_path(url)
    return p.exists() and p.stat().st_size > 0


def download_one(url):
    if is_cached(url):
        return url, None
    path = url_to_cache_path(url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = r.read()
        if not data:
            return url, "empty"
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.rename(path)
        return url, None
    except Exception as e:
        return url, str(e)


def download_many(urls, workers=16):
    CACHE_DIR.mkdir(exist_ok=True)
    todo = [u for u in urls if not is_cached(u)]
    cached = len(urls) - len(todo)
    print(f"  В кэше уже: {cached}. Нужно скачать: {len(todo)}")
    failed = set()
    if not todo:
        return failed
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(download_one, u) for u in todo]
            for f in as_completed(futures):
                url, err = f.result()
                if err:
                    failed.add(url)
                done += 1
                if done % 25 == 0 or done == len(todo):
                    print(f"  скачано {done}/{len(todo)}  ошибок: {len(failed)}",
                          end="\r", flush=True)
        print()
    except KeyboardInterrupt:
        print("\nПрерывание скачивания. Что успело — в кэше.")
    return failed


# ---------- сбор пар по категориям ----------

def collect_from_decisions(csv_path):
    """Читает pair_decisions.csv и возвращает {category: [(before_url, after_url), ...]}.
    Берёт ТОЛЬКО decision='good', оба URL должны быть изображениями по расширению,
    оба файла должны быть в кэше."""
    ALLOWED_URL_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".jfif", ".heic"}

    def is_image_url(url):
        ext = Path(url.split("?")[0]).suffix.lower()
        return ext in ALLOWED_URL_EXTS

    pairs_by_cat = defaultdict(list)
    skipped_decision = 0
    skipped_non_image = 0
    skipped_missing = 0
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("decision") != "good":
                skipped_decision += 1
                continue
            b, a = r["before_url"], r["after_url"]
            if not (is_image_url(b) and is_image_url(a)):
                skipped_non_image += 1
                continue
            if not (is_cached(b) and is_cached(a)):
                skipped_missing += 1
                continue
            pairs_by_cat[r["source"]].append((b, a))
    print(f"  Прочитано: {csv_path}")
    print(f"  Отброшено (decision != good):     {skipped_decision}")
    print(f"  Отброшено (URL не картинка):      {skipped_non_image}")
    print(f"  Отброшено (фото нет в кэше):      {skipped_missing}")
    return pairs_by_cat


def collect_from_xlsx(xlsx_paths, sample_per_cat, seed=42):
    if pd is None:
        print("Нужен pandas: pip install pandas openpyxl")
        sys.exit(1)
    random.seed(seed)
    ALLOWED = {"jpg", "jpeg", "png", "jfif", "webp", "heic", "heif", "bmp"}

    def ext(url):
        if not isinstance(url, str):
            return "NA"
        return url.split("?")[0].rsplit(".", 1)[-1].lower()

    pairs_by_cat = defaultdict(list)
    for xf in xlsx_paths:
        df = pd.read_excel(xf)
        cat = Path(xf).stem
        ok = df["Нарушение"].apply(ext).isin(ALLOWED) & df["Устранение"].apply(ext).isin(ALLOWED)
        clean = df[ok]
        for _, r in clean.iterrows():
            pairs_by_cat[cat].append((r["Нарушение"], r["Устранение"]))
        print(f"  {Path(xf).name}: {len(df)} → {len(clean)} после фильтра")

    sampled = {}
    for cat, pairs in pairs_by_cat.items():
        take = min(sample_per_cat, len(pairs))
        sampled[cat] = random.sample(pairs, take)
    return sampled


# ---------- generation negatives + split ----------

def build_rows(pairs_by_cat, failed_urls=None):
    if failed_urls is None:
        failed_urls = set()
    rows = []
    for cat, pairs in pairs_by_cat.items():
        good = [(b, a) for b, a in pairs
                if b not in failed_urls and a not in failed_urls
                and is_cached(b) and is_cached(a)]
        n = len(good)
        if n < 4:
            print(f"  {cat[:60]}: только {n} пар — пропускаю")
            continue
        for b, a in good:
            rows.append({
                "path_before": str(url_to_cache_path(b).resolve()),
                "path_after": str(url_to_cache_path(a).resolve()),
                "label": 1,
                "category": cat,
            })
        afters = [a for _, a in good]
        shuffled = afters[:]
        for _ in range(10):
            random.shuffle(shuffled)
            if all(shuffled[i] != afters[i] for i in range(n)):
                break
        for i, (b, _) in enumerate(good):
            rows.append({
                "path_before": str(url_to_cache_path(b).resolve()),
                "path_after": str(url_to_cache_path(shuffled[i]).resolve()),
                "label": 0,
                "category": cat,
            })
        print(f"  {cat[:60]}: {n} positive + {n} negative = {2*n}")
    return rows


def split_and_save(rows, train_csv, val_csv, val_split=0.2, seed=42):
    random.seed(seed)
    random.shuffle(rows)
    by_strat = defaultdict(list)
    for r in rows:
        by_strat[(r["category"], r["label"])].append(r)

    train, val = [], []
    for lst in by_strat.values():
        n_val = int(len(lst) * val_split)
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    random.shuffle(train)
    random.shuffle(val)

    print(f"  Train: {len(train)} "
          f"(pos {sum(1 for r in train if r['label']==1)}, "
          f"neg {sum(1 for r in train if r['label']==0)})")
    print(f"  Val:   {len(val)} "
          f"(pos {sum(1 for r in val if r['label']==1)}, "
          f"neg {sum(1 for r in val if r['label']==0)})")

    fields = ["path_before", "path_after", "label"]
    for name, data in [(train_csv, train), (val_csv, val)]:
        with open(name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in data:
                w.writerow({k: r[k] for k in fields})


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", nargs="*", help="xlsx файлы (для режима со скачиванием)")
    ap.add_argument("--from-decisions", default=None,
                    help="взять пары из pair_decisions.csv (БЕЗ скачивания)")
    ap.add_argument("--sample", type=int, default=5000,
                    help="(только для xlsx-режима) сколько positive pairs всего")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-csv", default="train.csv")
    ap.add_argument("--val-csv", default="val.csv")
    args = ap.parse_args()

    if args.from_decisions:
        print("=== Шаг 1/3: чтение pair_decisions.csv и проверка кэша ===")
        pairs_by_cat = collect_from_decisions(args.from_decisions)
        for cat, lst in pairs_by_cat.items():
            print(f"  {cat[:60]}: {len(lst)} готовых пар")
        if not pairs_by_cat:
            print("Нет ни одной пары с двумя фото в кэше.")
            sys.exit(1)

        print("\n=== Шаг 2/3: формирование positive + negative пар ===")
        rows = build_rows(pairs_by_cat, failed_urls=set())

        print(f"\nВсего: {len(rows)} пар "
              f"(pos {sum(1 for r in rows if r['label']==1)}, "
              f"neg {sum(1 for r in rows if r['label']==0)})")

        print(f"\n=== Шаг 3/3: train/val split ({1-args.val_split:.0%}/{args.val_split:.0%}) ===")
        split_and_save(rows, args.train_csv, args.val_csv,
                       val_split=args.val_split, seed=args.seed)

    else:
        if not args.xlsx:
            print("Нужно передать .xlsx файлы или использовать --from-decisions.")
            sys.exit(1)

        print("=== Шаг 1/5: чтение xlsx и фильтр расширений ===")
        n_per_cat = args.sample // len(args.xlsx)
        sampled = collect_from_xlsx(args.xlsx, n_per_cat, seed=args.seed)
        for cat, pairs in sampled.items():
            print(f"  {cat[:60]}: выбрано {len(pairs)} пар")

        print(f"\n=== Шаг 2/5: скачивание фото в {CACHE_DIR}/ ===")
        all_urls = list({u for pairs in sampled.values() for pair in pairs for u in pair})
        print(f"  Уникальных URL: {len(all_urls)}")
        failed = download_many(all_urls, workers=args.workers)
        if failed:
            print(f"  Не скачано: {len(failed)} (пары с ними пропустятся)")

        print("\n=== Шаг 3/5: формирование positive + negative пар ===")
        rows = build_rows(sampled, failed_urls=failed)

        print(f"\nВсего: {len(rows)} пар "
              f"(pos {sum(1 for r in rows if r['label']==1)}, "
              f"neg {sum(1 for r in rows if r['label']==0)})")

        print(f"\n=== Шаг 4/5: train/val split ({1-args.val_split:.0%}/{args.val_split:.0%}) ===")
        split_and_save(rows, args.train_csv, args.val_csv,
                       val_split=args.val_split, seed=args.seed)

    print(f"\nГотово!")
    print(f"  Создано: {args.train_csv}, {args.val_csv}")
    print(f"\nДальше — обучение:")
    print(f"  python3 train_siamese.py --train-csv {args.train_csv} --val-csv {args.val_csv} --epochs 10")


if __name__ == "__main__":
    main()
