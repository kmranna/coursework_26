#!/usr/bin/env python3
"""
Автокорректор pair_decisions.csv.

Идея: bad-пары часто были помечены пока фото ещё не успело скачаться.
Теперь они в кэше. Проверяем — если оба файла открываются как изображения,
переводим decision из 'bad' в 'good'. Если хотя бы одно — документ (PDF/DOCX/...)
или ломаное, оставляем bad.

Запуск:
    python3 fix_bad_decisions.py
    python3 fix_bad_decisions.py --dry-run    # только показать что будет изменено
"""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

CACHE_DIR = Path("image_cache")
CSV_PATH = Path("pair_decisions.csv")
BACKUP_PATH = Path("pair_decisions.csv.bak")


def url_to_cache_path(url):
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        ext = ".jpg"
    return CACHE_DIR / f"{h}{ext}"


def file_is_image(path):
    """True если файл существует и PIL может его открыть."""
    if not path.exists() or path.stat().st_size < 100:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        # verify() закрывает файл, для load() нужно открыть заново
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="не сохранять изменения, только показать сводку")
    args = ap.parse_args()

    if not CSV_PATH.exists():
        print(f"Нет файла {CSV_PATH}")
        return

    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fields = list(rows[0].keys())
    bads = [r for r in rows if r["decision"] == "bad"]
    print(f"Bad пар: {len(bads)}")
    print("Проверяю каждую...\n")

    n_to_good = 0
    n_keep_bad = 0
    examples = []

    for i, r in enumerate(bads):
        b_path = url_to_cache_path(r["before_url"])
        a_path = url_to_cache_path(r["after_url"])
        b_ok = file_is_image(b_path)
        a_ok = file_is_image(a_path)
        if b_ok and a_ok:
            r["decision"] = "good"
            n_to_good += 1
            if len(examples) < 5:
                examples.append(("→ good", r["source"][:25], r["row"]))
        else:
            n_keep_bad += 1
            if len([e for e in examples if e[0] == "остался bad"]) < 5:
                reason = []
                if not b_ok:
                    reason.append("before не открылось")
                if not a_ok:
                    reason.append("after не открылось")
                examples.append(("остался bad", r["source"][:25], r["row"]))

        if (i + 1) % 20 == 0:
            print(f"  обработано {i+1}/{len(bads)}", end="\r", flush=True)
    print()

    print(f"\nИтог:")
    print(f"  Возвращено в 'good': {n_to_good}")
    print(f"  Оставлено 'bad':     {n_keep_bad}")

    print(f"\nПримеры:")
    for tag, src, row in examples[:10]:
        print(f"  {tag:12s}  {src}  row {row}")

    new_good = sum(1 for r in rows if r["decision"] == "good")
    new_bad = sum(1 for r in rows if r["decision"] == "bad")
    print(f"\nИтоговое распределение в pair_decisions.csv:")
    print(f"  good: {new_good}")
    print(f"  bad:  {new_bad}")

    if args.dry_run:
        print("\n[dry-run] Файл не сохранён. Уберите --dry-run чтобы записать изменения.")
        return

    # backup и сохранение
    shutil.copy2(CSV_PATH, BACKUP_PATH)
    print(f"\nBackup: {BACKUP_PATH}")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Сохранено: {CSV_PATH}")


if __name__ == "__main__":
    main()
