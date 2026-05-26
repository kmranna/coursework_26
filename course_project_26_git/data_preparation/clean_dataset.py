#!/usr/bin/env python3
"""
Автофильтр: убирает из xlsx пары, где в колонке «Устранение» лежит не фото,
а документ (PDF / DOCX / XLSX и т.п.) — на основе разметки 500 ручных пар
такие пары всегда оказывались плохими.

Запуск:
    python clean_dataset.py file1.xlsx [file2.xlsx ...]

Создаёт:
    clean_dataset.csv     — чистые пары для обучения модели
    rejected_dataset.csv  — отброшенные строки (для проверки)
    summary.txt           — сводная статистика
"""

import sys
import csv
from pathlib import Path
from collections import Counter

import pandas as pd

# Разрешённые расширения фото для колонки «после»
ALLOWED_EXTS = {"jpg", "jpeg", "png", "jfif", "webp", "heic", "heif", "bmp", "tif", "tiff"}


def url_ext(url: str) -> str:
    if not isinstance(url, str):
        return "NA"
    return url.split("?")[0].rsplit(".", 1)[-1].lower()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    all_rows = []
    summary_lines = []

    for xf in sys.argv[1:]:
        xp = Path(xf)
        if not xp.exists():
            print(f"Не найден: {xp}")
            continue
        df = pd.read_excel(xp)
        df["source"] = xp.stem
        df["row_in_source"] = df.index
        df["ext_after"] = df["Устранение"].apply(url_ext)
        all_rows.append(df)

        line = f"{xp.name}: {len(df)} строк"
        summary_lines.append(line)
        print(line)

    if not all_rows:
        print("Нет данных.")
        sys.exit(1)

    full = pd.concat(all_rows, ignore_index=True)
    print(f"\nИТОГО: {len(full)} строк во всех файлах")

    # Распределение расширений в «после»
    print("\nРасширения в «Устранение»:")
    summary_lines.append("\nРасширения в 'Устранение':")
    for k, v in Counter(full["ext_after"]).most_common():
        line = f"  {k:6s}: {v:6d} ({100*v/len(full):5.2f}%)"
        print(line)
        summary_lines.append(line)

    # Фильтр
    mask_clean = full["ext_after"].isin(ALLOWED_EXTS)
    clean = full[mask_clean].copy()
    rejected = full[~mask_clean].copy()

    rejected_reason = rejected["ext_after"].map(
        lambda e: f"after_is_{e}" if e != "NA" else "after_missing"
    )
    rejected.insert(0, "reject_reason", rejected_reason)

    print(f"\nЧистых пар: {len(clean)} ({100*len(clean)/len(full):.2f}%)")
    print(f"Отброшено:  {len(rejected)} ({100*len(rejected)/len(full):.2f}%)")
    summary_lines.append(f"\nЧистых пар: {len(clean)} ({100*len(clean)/len(full):.2f}%)")
    summary_lines.append(f"Отброшено:  {len(rejected)} ({100*len(rejected)/len(full):.2f}%)")

    # Распределение источников в результатах
    print("\nЧистые пары по источникам:")
    summary_lines.append("\nЧистые пары по источникам:")
    for s, n in clean["source"].value_counts().items():
        line = f"  {s}: {n}"
        print(line)
        summary_lines.append(line)

    # Сохранить
    out_cols = ["source", "row_in_source", "Название задачи", "Нарушение", "Устранение"]
    clean[out_cols].to_csv("clean_dataset.csv", index=False, encoding="utf-8")
    rejected_cols = ["reject_reason"] + out_cols + ["ext_after"]
    rejected[rejected_cols].to_csv("rejected_dataset.csv", index=False, encoding="utf-8")

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\nСохранено:")
    print("  clean_dataset.csv     — для обучения модели")
    print("  rejected_dataset.csv  — отброшенные строки (для проверки)")
    print("  summary.txt           — сводка")


if __name__ == "__main__":
    main()
