#!/usr/bin/env python3
"""
GUI для верификации подозрительных пар найденных CLIP-детектором.

Читает suspicious.csv (от detect_anomalies.py), для каждой пары показывает оба фото,
CLIP-скоры и предсказанную категорию. По решениям обновляет pair_decisions.csv.

Кнопки:
  1 / ←   нормальное фото (false positive) → decision='good'
  2 / →   аномалия подтверждена         → decision='bad'
  3 / ↓   не уверен                     → пропустить
  B / ↑   назад
  Q       выйти (прогресс сохранён)

Запуск:
    python3 review_suspicious.py
    python3 review_suspicious.py --csv suspicious.csv
"""

import argparse
import csv
import hashlib
from pathlib import Path
import tkinter as tk

from PIL import Image, ImageTk, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

CACHE_DIR = Path("image_cache")
DECISIONS_CSV = Path("pair_decisions.csv")


def url_to_cache_path(url):
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        ext = ".jpg"
    return CACHE_DIR / f"{h}{ext}"


def decision_key(source, row):
    return f"{source}#{row}"


def load_decisions():
    d = {}
    if DECISIONS_CSV.exists():
        with open(DECISIONS_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d[decision_key(row["source"], int(row["row"]))] = row
    return d


def save_decisions(decisions):
    fields = ["source", "row", "task", "before_url", "after_url", "decision"]
    with open(DECISIONS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in decisions.values():
            w.writerow({k: v.get(k, "") for k in fields})


class ReviewApp:
    THUMB = (520, 600)

    def __init__(self, root, suspicious_rows, decisions):
        self.root = root
        self.decisions = decisions
        self.pairs = []
        missing = 0
        for r in suspicious_rows:
            key = decision_key(r["source"], int(r["row"]))
            dec = decisions.get(key)
            if not dec:
                missing += 1
                continue
            self.pairs.append({
                "key": key,
                "source": r["source"],
                "row": int(r["row"]),
                "task": dec.get("task", ""),
                "before_url": dec["before_url"],
                "after_url": dec["after_url"],
                "category": r["predicted_category"],
                "real": float(r["score_real_photo"]),
                "ui": float(r["score_ui_screenshot"]),
                "doc": float(r["score_document_photo"]),
                "sign": float(r["score_signage_photo"]),
            })
        if missing:
            print(f"Внимание: {missing} строк из suspicious.csv не нашлись в pair_decisions.csv")
        self.idx = 0

        root.title("Верификация подозрительных пар")
        root.geometry("1200x900")
        root.configure(bg="#1e1e1e")

        self.info_label = tk.Label(root, text="", font=("Arial", 13, "bold"),
                                    bg="#1e1e1e", fg="white", pady=4)
        self.info_label.pack()

        self.score_label = tk.Label(root, text="", font=("Arial", 11),
                                    bg="#1e1e1e", fg="#ffaa00")
        self.score_label.pack()

        imgs_frame = tk.Frame(root, bg="#1e1e1e")
        imgs_frame.pack(pady=8, expand=True)

        before_box = tk.Frame(imgs_frame, bg="#1e1e1e")
        before_box.pack(side="left", padx=10)
        tk.Label(before_box, text="ДО (нарушение)", font=("Arial", 11, "bold"),
                 bg="#1e1e1e", fg="#4caf50").pack()
        self.before_img = tk.Label(before_box, bg="#1e1e1e")
        self.before_img.pack()

        after_box = tk.Frame(imgs_frame, bg="#1e1e1e")
        after_box.pack(side="left", padx=10)
        tk.Label(after_box, text="ПОСЛЕ — ПОДОЗРИТЕЛЬНОЕ",
                 font=("Arial", 11, "bold"), bg="#1e1e1e", fg="#f44336").pack()
        self.after_img = tk.Label(after_box, bg="#1e1e1e")
        self.after_img.pack()

        self.decision_label = tk.Label(root, text="", font=("Arial", 12, "italic"),
                                       bg="#1e1e1e", fg="#aaaaaa")
        self.decision_label.pack(pady=4)

        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="✓ Нормальное фото  [1 / ←]", bg="#4caf50", fg="white",
                  font=("Arial", 12, "bold"), width=24, height=2, bd=0,
                  command=lambda: self.mark("good")).pack(side="left", padx=6)
        tk.Button(btn_frame, text="✗ Аномалия  [2 / →]", bg="#f44336", fg="white",
                  font=("Arial", 12, "bold"), width=22, height=2, bd=0,
                  command=lambda: self.mark("bad")).pack(side="left", padx=6)
        tk.Button(btn_frame, text="? Не уверен  [3 / ↓]", bg="#ff9800", fg="white",
                  font=("Arial", 12, "bold"), width=20, height=2, bd=0,
                  command=lambda: self.mark("unsure")).pack(side="left", padx=6)

        for key, fn in [
            ("1", lambda e: self.mark("good")),
            ("<Left>", lambda e: self.mark("good")),
            ("2", lambda e: self.mark("bad")),
            ("<Right>", lambda e: self.mark("bad")),
            ("3", lambda e: self.mark("unsure")),
            ("<Down>", lambda e: self.mark("unsure")),
            ("b", lambda e: self.go_back()),
            ("B", lambda e: self.go_back()),
            ("<Up>", lambda e: self.go_back()),
            ("q", lambda e: self.quit()),
            ("Q", lambda e: self.quit()),
        ]:
            root.bind(key, fn)

        if self.pairs:
            self.show_current()

    def _load_image(self, path):
        try:
            img = Image.open(path)
            w, h = img.size
            img.thumbnail(self.THUMB)
            photo = ImageTk.PhotoImage(img)
            return photo, f"{w}×{h}"
        except Exception as e:
            return None, str(e)

    def _set_image(self, label, path):
        if path.exists():
            photo, _ = self._load_image(path)
            if photo:
                label.config(image=photo, text="", width=0, height=0)
                label.image = photo
                return
        label.config(image="", text="(не открывается)", fg="red", width=50, height=15)
        label.image = None

    def show_current(self):
        if not self.pairs:
            return
        p = self.pairs[self.idx]
        self._set_image(self.before_img, url_to_cache_path(p["before_url"]))
        self._set_image(self.after_img, url_to_cache_path(p["after_url"]))

        self.info_label.config(
            text=f"[{self.idx+1}/{len(self.pairs)}]  {p['source']} row {p['row']}  •  {p['task']}"
        )
        self.score_label.config(
            text=f"CLIP: real={p['real']:.1f}  ui={p['ui']:.1f}  "
                 f"doc={p['doc']:.1f}  sign={p['sign']:.1f}  →  предсказано: {p['category']}"
        )
        cur = self.decisions.get(p["key"], {}).get("decision", "—")
        labels = {"good": "✓ good", "bad": "✗ bad", "unsure": "? unsure"}
        self.decision_label.config(text=f"Текущее решение в pair_decisions: {labels.get(cur, cur)}")

    def mark(self, decision):
        if not self.pairs:
            return
        p = self.pairs[self.idx]
        if p["key"] in self.decisions:
            self.decisions[p["key"]]["decision"] = decision
            save_decisions(self.decisions)
        if self.idx < len(self.pairs) - 1:
            self.idx += 1
            self.show_current()
        else:
            # finished
            self.info_label.config(text="Готово! Все 132 пар просмотрены.")
            self.score_label.config(text="Можно закрывать (Q или крестик).")

    def go_back(self):
        if self.idx > 0:
            self.idx -= 1
            self.show_current()

    def quit(self):
        save_decisions(self.decisions)
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="suspicious.csv")
    args = ap.parse_args()

    if not Path(args.csv).exists():
        print(f"Не найден {args.csv}. Сначала запустите detect_anomalies.py")
        return

    decisions = load_decisions()
    print(f"Загружено решений из {DECISIONS_CSV}: {len(decisions)}")

    with open(args.csv, encoding="utf-8") as f:
        suspicious_rows = list(csv.DictReader(f))
    print(f"Подозрительных в {args.csv}: {len(suspicious_rows)}")

    # сводка ДО ревью
    bad_before = sum(1 for v in decisions.values() if v["decision"] == "bad")
    print(f"\nДо ревью: bad = {bad_before}\n")

    root = tk.Tk()
    app = ReviewApp(root, suspicious_rows, decisions)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()

    decisions_after = load_decisions()
    bad_after = sum(1 for v in decisions_after.values() if v["decision"] == "bad")
    good_after = sum(1 for v in decisions_after.values() if v["decision"] == "good")
    print(f"\nПосле ревью: good {good_after}, bad {bad_after}")
    print(f"Добавлено в bad: {bad_after - bad_before}")


if __name__ == "__main__":
    main()
