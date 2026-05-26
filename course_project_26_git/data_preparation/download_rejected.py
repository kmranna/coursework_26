#!/usr/bin/env python3
"""
Скачивает все отброшенные файлы (PDF, DOCX, XLSX и т.п.) из rejected_dataset.csv
в папку rejected_files/, чтобы их можно было пролистать в Finder.

Запуск:
    python3 download_rejected.py
"""

import csv
import sys
import urllib.request
import urllib.error
import socket
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CSV_PATH = Path("rejected_dataset.csv")
OUT_DIR = Path("rejected_files")
TIMEOUT = 20


def download_one(url, dst):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = r.read()
        dst.write_bytes(data)
        return url, None
    except Exception as e:
        return url, str(e)


def main():
    if not CSV_PATH.exists():
        print(f"Не найден {CSV_PATH}. Сначала запустите clean_dataset.py")
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)

    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    print(f"Найдено отброшенных пар: {len(rows)}")
    print(f"Скачиваю файлы 'Устранение' в {OUT_DIR}/ ...")

    jobs = []
    for r in rows:
        url = r["Устранение"]
        ext = r["ext_after"]
        # понятное имя файла: причина_источник_строка.расширение
        # обрезаем длинное имя источника
        src_short = r["source"].replace("Завершенные_задачи__", "").strip("_")[:25]
        name = f"{r['reject_reason']}_{src_short}_row{r['row_in_source']}.{ext}"
        # на всякий случай уберём пробелы и подозрительные символы
        name = name.replace("/", "_").replace(" ", "_")
        dst = OUT_DIR / name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        jobs.append((url, dst))

    print(f"К скачиванию: {len(jobs)} (остальные уже в кэше)")

    if not jobs:
        print("Всё уже скачано.")
        print(f"\nОткрыть папку: open {OUT_DIR}")
        return

    errors = []
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(download_one, url, dst) for url, dst in jobs]
        for f in as_completed(futures):
            url, err = f.result()
            if err:
                errors.append((url, err))
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"  скачано {done}/{len(jobs)}  ошибок: {len(errors)}",
                      end="\r", flush=True)
    print()

    if errors:
        print(f"\nОшибок скачивания: {len(errors)}")
        for url, e in errors[:5]:
            print(f"  {e}: {url}")

    print(f"\nГотово. Откройте папку:")
    print(f"  open {OUT_DIR}")


if __name__ == "__main__":
    main()
