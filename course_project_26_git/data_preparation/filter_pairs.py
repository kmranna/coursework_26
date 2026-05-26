#!/usr/bin/env python3
"""
Фильтр пар фото «до / после» для задач благоустройства.

Читает xlsx с колонками: «Название задачи», «Нарушение» (URL до), «Устранение» (URL после).
Скачивает фото в локальный кэш, потом показывает пары в окне для ручной разметки:
годится ли пара для обучения модели «исправлено / не исправлено».

Запуск:
    python filter_pairs.py file1.xlsx [file2.xlsx ...]
        [--sample N]        # размечать только N случайных пар (по умолчанию: все)
        [--seed S]          # сид для выборки (по умолчанию: 42)
        [--workers K]       # параллельных потоков скачивания (по умолчанию: 16)
        [--skip-download]   # не скачивать, использовать только что есть в кэше
        [--no-download]     # синоним --skip-download

Горячие клавиши в окне:
    1 или ←   — пара ОК (оба фото реальные)
    2 или →   — пара плохая (хотя бы одно — скриншот системы)
    3 или ↓   — не уверен
    B или ↑   — назад
    O         — открыть оба фото в системном просмотрщике
    Q         — выйти (прогресс сохранён)

Результат:
    image_cache/             — скачанные фото (по MD5 от URL)
    pair_decisions.csv       — таблица решений (source, row, before_url, after_url, decision)
    good_pairs/, bad_pairs/  — копии фото, разложенные по решениям (создаются при выходе)
"""

import sys
import csv
import hashlib
import argparse
import subprocess
import urllib.request
import urllib.error
import socket
import shutil
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import pandas as pd
except ImportError:
    print("Нужен pandas: pip install pandas openpyxl")
    sys.exit(1)

try:
    import tkinter as tk
    from PIL import Image, ImageTk
except ImportError:
    print("Нужны tkinter и Pillow: pip install Pillow  (tkinter обычно идёт с Python)")
    sys.exit(1)


CACHE_DIR = Path("image_cache")
DECISIONS_CSV = Path("pair_decisions.csv")
FAILED_LOG = Path("download_failures.csv")
GOOD_DIR = Path("good_pairs")
BAD_DIR = Path("bad_pairs")

USER_AGENT = "Mozilla/5.0 (filter_pairs.py)"
TIMEOUT = 20


# --------------------------- скачивание ---------------------------

def url_to_cache_path(url: str) -> Path:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        ext = ".jpg"
    return CACHE_DIR / f"{h}{ext}"


def is_cached(url: str) -> bool:
    p = url_to_cache_path(url)
    return p.exists() and p.stat().st_size > 0


def download_one(url: str):
    path = url_to_cache_path(url)
    if is_cached(url):
        return url, path, None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = r.read()
        if not data:
            return url, path, "empty"
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.rename(path)
        return url, path, None
    except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout) as e:
        return url, path, str(e)
    except Exception as e:
        return url, path, f"{type(e).__name__}: {e}"


def download_all(urls, workers=16):
    CACHE_DIR.mkdir(exist_ok=True)
    todo = [u for u in urls if not is_cached(u)]
    cached = len(urls) - len(todo)
    print(f"Кэш: {cached} уже есть, нужно скачать {len(todo)}")
    if not todo:
        return {}

    failures = {}
    done = 0
    total = len(todo)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, u): u for u in todo}
        try:
            for f in as_completed(futures):
                url, _, err = f.result()
                if err:
                    failures[url] = err
                done += 1
                if done % 25 == 0 or done == total:
                    pct = 100 * done / total
                    print(f"  скачано {done}/{total} ({pct:.1f}%)  ошибок: {len(failures)}",
                          end="\r", flush=True)
        except KeyboardInterrupt:
            print("\nПрерывание скачивания. Что успело — сохранено в кэше.")
    print()

    if failures:
        with open(FAILED_LOG, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["url", "error"])
            for u, e in failures.items():
                w.writerow([u, e])
        print(f"Записаны ошибки скачивания: {FAILED_LOG} ({len(failures)} шт.)")
    return failures


# --------------------------- решения ---------------------------

def decision_key(source: str, row: int) -> str:
    return f"{source}#{row}"


def load_decisions() -> dict[str, dict]:
    d = {}
    if DECISIONS_CSV.exists():
        with open(DECISIONS_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d[decision_key(row["source"], int(row["row"]))] = row
    return d


def save_decisions(decisions: dict[str, dict]) -> None:
    fields = ["source", "row", "task", "before_url", "after_url", "decision"]
    with open(DECISIONS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in decisions.values():
            w.writerow({k: v.get(k, "") for k in fields})


def export_by_decision(pairs, decisions):
    GOOD_DIR.mkdir(exist_ok=True)
    BAD_DIR.mkdir(exist_ok=True)
    for p in pairs:
        key = decision_key(p["source"], p["row"])
        rec = decisions.get(key)
        if not rec:
            continue
        target = {"good": GOOD_DIR, "bad": BAD_DIR}.get(rec["decision"])
        if not target:
            continue
        for tag, url in (("before", p["before_url"]), ("after", p["after_url"])):
            src = url_to_cache_path(url)
            if not src.exists():
                continue
            name = f"{p['source']}_row{p['row']}_{tag}{src.suffix}"
            try:
                shutil.copy2(src, target / name)
            except Exception:
                pass


# --------------------------- GUI ---------------------------

class PairFilterApp:
    THUMB_SIZE = (520, 600)

    def __init__(self, root: tk.Tk, pairs: list[dict]):
        self.root = root
        self.pairs = pairs
        self.decisions = load_decisions()
        self.idx = 0

        root.title("Фильтр пар фото")
        root.geometry("1200x900")
        root.configure(bg="#1e1e1e")

        self.info_label = tk.Label(
            root, text="", font=("Arial", 13, "bold"),
            bg="#1e1e1e", fg="white", pady=6,
        )
        self.info_label.pack()

        self.stats_label = tk.Label(
            root, text="", font=("Arial", 10),
            bg="#1e1e1e", fg="#aaaaaa",
        )
        self.stats_label.pack()

        # фото бок о бок
        imgs_frame = tk.Frame(root, bg="#1e1e1e")
        imgs_frame.pack(pady=8, expand=True)

        before_box = tk.Frame(imgs_frame, bg="#1e1e1e")
        before_box.pack(side="left", padx=10)
        tk.Label(before_box, text="ДО (нарушение)", font=("Arial", 11, "bold"),
                 bg="#1e1e1e", fg="#4caf50").pack()
        self.before_img = tk.Label(before_box, bg="#1e1e1e")
        self.before_img.pack()
        self.before_meta = tk.Label(before_box, text="", font=("Arial", 9),
                                    bg="#1e1e1e", fg="#888888")
        self.before_meta.pack()

        after_box = tk.Frame(imgs_frame, bg="#1e1e1e")
        after_box.pack(side="left", padx=10)
        tk.Label(after_box, text="ПОСЛЕ (устранение)", font=("Arial", 11, "bold"),
                 bg="#1e1e1e", fg="#2196f3").pack()
        self.after_img = tk.Label(after_box, bg="#1e1e1e")
        self.after_img.pack()
        self.after_meta = tk.Label(after_box, text="", font=("Arial", 9),
                                   bg="#1e1e1e", fg="#888888")
        self.after_meta.pack()

        self.decision_label = tk.Label(
            root, text="", font=("Arial", 12, "italic"),
            bg="#1e1e1e", fg="#ffaa00",
        )
        self.decision_label.pack(pady=4)

        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="✓ Пара ОК  [1 / ←]", bg="#4caf50", fg="white",
                  font=("Arial", 12, "bold"), width=20, height=2, bd=0,
                  command=lambda: self.mark("good")).pack(side="left", padx=6)
        tk.Button(btn_frame, text="✗ Есть скриншот  [2 / →]", bg="#f44336", fg="white",
                  font=("Arial", 12, "bold"), width=22, height=2, bd=0,
                  command=lambda: self.mark("bad")).pack(side="left", padx=6)
        tk.Button(btn_frame, text="? Не уверен  [3 / ↓]", bg="#ff9800", fg="white",
                  font=("Arial", 12, "bold"), width=20, height=2, bd=0,
                  command=lambda: self.mark("unsure")).pack(side="left", padx=6)

        nav_frame = tk.Frame(root, bg="#1e1e1e")
        nav_frame.pack(pady=4)
        tk.Button(nav_frame, text="← Назад  [B / ↑]", bg="#555555", fg="white",
                  font=("Arial", 10), width=16, command=self.go_back).pack(side="left", padx=4)
        tk.Button(nav_frame, text="🔍 Открыть полностью  [O]", bg="#555555", fg="white",
                  font=("Arial", 10), width=22, command=self.open_external).pack(side="left", padx=4)

        for keysym, fn in [
            ("1", lambda e: self.mark("good")),
            ("<Left>", lambda e: self.mark("good")),
            ("2", lambda e: self.mark("bad")),
            ("<Right>", lambda e: self.mark("bad")),
            ("3", lambda e: self.mark("unsure")),
            ("<Down>", lambda e: self.mark("unsure")),
            ("b", lambda e: self.go_back()),
            ("B", lambda e: self.go_back()),
            ("<Up>", lambda e: self.go_back()),
            ("o", lambda e: self.open_external()),
            ("O", lambda e: self.open_external()),
            ("q", lambda e: self.quit()),
            ("Q", lambda e: self.quit()),
        ]:
            root.bind(keysym, fn)

        # начинаем с первой неразмеченной
        for i, p in enumerate(self.pairs):
            if decision_key(p["source"], p["row"]) not in self.decisions:
                self.idx = i
                break

        self.show_current()

    def _load_image(self, path: Path):
        try:
            img = Image.open(path)
            w, h = img.size
            img.thumbnail(self.THUMB_SIZE)
            photo = ImageTk.PhotoImage(img)
            return photo, f"{path.name}  ({w}×{h}, {path.stat().st_size // 1024} КБ)"
        except Exception as e:
            return None, f"Ошибка: {e}"

    def _set_image_or_placeholder(self, img_label, meta_label, path, url):
        """Показывает картинку, либо placeholder если файла нет.
        Важно: при показе картинки сбрасываем width/height (иначе после placeholder
        Label остаётся со старыми размерами в символах и картинка показывается мелко)."""
        if path.exists():
            photo, meta = self._load_image(path)
            if photo:
                # width=0/height=0 → Label адаптируется под размер изображения
                img_label.config(image=photo, text="", width=0, height=0)
                img_label.image = photo
                meta_label.config(text=meta)
                return
        # сюда попадаем если файла нет ИЛИ не удалось загрузить
        img_label.config(image="", text="(не скачано)", fg="red", width=60, height=20)
        img_label.image = None
        meta_label.config(text=url[:80])

    def show_current(self):
        if not self.pairs:
            self.info_label.config(text="Пар нет.")
            return

        self.idx = max(0, min(self.idx, len(self.pairs) - 1))
        p = self.pairs[self.idx]

        before_path = url_to_cache_path(p["before_url"])
        after_path = url_to_cache_path(p["after_url"])

        self._set_image_or_placeholder(self.before_img, self.before_meta,
                                       before_path, p["before_url"])
        self._set_image_or_placeholder(self.after_img, self.after_meta,
                                       after_path, p["after_url"])

        self.info_label.config(
            text=f"[{self.idx + 1}/{len(self.pairs)}]  {p['source']} стр. {p['row']}  •  {p['task']}"
        )

        good = sum(1 for v in self.decisions.values() if v["decision"] == "good")
        bad = sum(1 for v in self.decisions.values() if v["decision"] == "bad")
        unsure = sum(1 for v in self.decisions.values() if v["decision"] == "unsure")
        done = good + bad + unsure
        self.stats_label.config(
            text=f"Размечено: {done}/{len(self.pairs)}  •  ✓ {good}  •  ✗ {bad}  •  ? {unsure}"
        )

        cur = self.decisions.get(decision_key(p["source"], p["row"]))
        if cur:
            labels = {"good": "✓ ОК", "bad": "✗ есть скриншот", "unsure": "? не уверен"}
            self.decision_label.config(text=f"Текущее решение: {labels.get(cur['decision'])}")
        else:
            self.decision_label.config(text="Не размечено")

    def mark(self, decision: str):
        if not self.pairs:
            return
        p = self.pairs[self.idx]
        key = decision_key(p["source"], p["row"])
        self.decisions[key] = {
            "source": p["source"],
            "row": p["row"],
            "task": p["task"],
            "before_url": p["before_url"],
            "after_url": p["after_url"],
            "decision": decision,
        }
        save_decisions(self.decisions)
        if self.idx < len(self.pairs) - 1:
            self.idx += 1
        self.show_current()

    def go_back(self):
        if self.idx > 0:
            self.idx -= 1
            self.show_current()

    def open_external(self):
        if not self.pairs:
            return
        p = self.pairs[self.idx]
        for url in (p["before_url"], p["after_url"]):
            path = url_to_cache_path(url)
            if not path.exists():
                continue
            try:
                if sys.platform.startswith("darwin"):
                    subprocess.run(["open", str(path)])
                elif sys.platform.startswith("win"):
                    subprocess.run(["cmd", "/c", "start", "", str(path)], shell=False)
                else:
                    subprocess.run(["xdg-open", str(path)])
            except Exception:
                pass

    def quit(self):
        save_decisions(self.decisions)
        self.root.destroy()


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Фильтр пар фото для задач благоустройства.")
    ap.add_argument("xlsx", nargs="*", help="один или несколько .xlsx файлов (не нужны при --review-bad/--review-all)")
    ap.add_argument("--sample", type=int, default=None,
                    help="взять только N случайных пар (по умолчанию — все)")
    ap.add_argument("--per-source", type=int, default=None,
                    help="взять N НОВЫХ пар из каждого xlsx (уже размеченные сохраняются дополнительно)")
    ap.add_argument("--review-bad", action="store_true",
                    help="режим пересмотра: показать только пары, размеченные как 'bad', чтобы поменять решения")
    ap.add_argument("--review-all", action="store_true",
                    help="режим пересмотра: показать все уже размеченные пары")
    ap.add_argument("--seed", type=int, default=42, help="сид для выборки")
    ap.add_argument("--workers", type=int, default=16, help="потоков скачивания")
    ap.add_argument("--skip-download", "--no-download", action="store_true",
                    help="не скачивать, использовать только кэш")
    args = ap.parse_args()

    # 0. режим пересмотра — строим pairs прямо из pair_decisions.csv, xlsx не нужны
    if args.review_bad or args.review_all:
        existing_decisions = load_decisions()
        if not existing_decisions:
            print(f"Не найдено решений в {DECISIONS_CSV}")
            sys.exit(1)
        wanted = {"bad"} if args.review_bad else {"good", "bad", "unsure"}
        pairs = []
        for rec in existing_decisions.values():
            if rec["decision"] not in wanted:
                continue
            pairs.append({
                "source": rec["source"],
                "row": int(rec["row"]),
                "task": rec["task"],
                "before_url": rec["before_url"],
                "after_url": rec["after_url"],
            })
        if not pairs:
            print(f"Нет пар с нужным решением ({wanted}).")
            sys.exit(0)
        print(f"Режим пересмотра: показываю {len(pairs)} пар (decision in {wanted})")

        # картинки должны быть в кэше; если чего-то не хватает — докачаем
        if not args.skip_download:
            urls = []
            for p in pairs:
                if not is_cached(p["before_url"]):
                    urls.append(p["before_url"])
                if not is_cached(p["after_url"]):
                    urls.append(p["after_url"])
            if urls:
                print(f"Докачиваю {len(urls)} недостающих фото...")
                download_all(list(dict.fromkeys(urls)), workers=args.workers)

        print("\nОткрываю окно. Жмите 1 (good) на парах которые были помечены ошибочно.")
        print("Прогресс автосохраняется. Q — выйти.\n")
        root = tk.Tk()
        app = PairFilterApp(root, pairs)
        # начать с первой пары, а не с первой "неразмеченной"
        app.idx = 0
        app.show_current()
        root.protocol("WM_DELETE_WINDOW", app.quit)
        root.mainloop()

        # после ревью — пересчитаем сводку
        decisions = load_decisions()
        good = sum(1 for v in decisions.values() if v["decision"] == "good")
        bad = sum(1 for v in decisions.values() if v["decision"] == "bad")
        unsure = sum(1 for v in decisions.values() if v["decision"] == "unsure")
        print(f"\nПосле ревью: ✓ {good}  ✗ {bad}  ? {unsure}")
        return

    # обычный режим — нужны xlsx
    if not args.xlsx:
        print("Нужно передать хотя бы один .xlsx файл (или используйте --review-bad).")
        sys.exit(1)

    # 1. собрать пары из всех xlsx (отдельно по источникам)
    existing_decisions = load_decisions()
    pairs_by_source: dict[str, list[dict]] = {}

    for xf in args.xlsx:
        xp = Path(xf)
        if not xp.exists():
            print(f"Не найден файл: {xp}")
            sys.exit(1)
        print(f"Читаю {xp.name}...")
        df = pd.read_excel(xp)

        # ожидаем колонки «Название задачи», «Нарушение», «Устранение»
        cols = {c.strip(): c for c in df.columns}
        col_task = cols.get("Название задачи") or list(df.columns)[0]
        col_before = cols.get("Нарушение") or list(df.columns)[1]
        col_after = cols.get("Устранение") or list(df.columns)[2]

        src_pairs = []
        for i, row in df.iterrows():
            before, after = row[col_before], row[col_after]
            if not isinstance(before, str) or not isinstance(after, str):
                continue
            if not before.startswith("http") or not after.startswith("http"):
                continue
            src_pairs.append({
                "source": xp.stem,
                "row": int(i),
                "task": str(row[col_task])[:80],
                "before_url": before,
                "after_url": after,
            })
        pairs_by_source[xp.stem] = src_pairs
        labeled_in_src = sum(
            1 for p in src_pairs
            if decision_key(p["source"], p["row"]) in existing_decisions
        )
        print(f"  пар: {len(src_pairs)} (уже размечено: {labeled_in_src})")

    # 2. сэмплирование
    pairs: list[dict] = []
    if args.per_source:
        # per-source: всегда включаем уже размеченные из источника + N новых случайных
        random.seed(args.seed)
        for source, src_pairs in pairs_by_source.items():
            labeled = [p for p in src_pairs
                       if decision_key(p["source"], p["row"]) in existing_decisions]
            unlabeled = [p for p in src_pairs
                         if decision_key(p["source"], p["row"]) not in existing_decisions]
            take_new = min(args.per_source, len(unlabeled))
            new_sample = random.sample(unlabeled, take_new) if take_new else []
            picked = labeled + new_sample
            pairs.extend(picked)
            print(f"  {source}: {len(labeled)} уже размечено + {len(new_sample)} новых = {len(picked)}")
    else:
        for src_pairs in pairs_by_source.values():
            pairs.extend(src_pairs)
        if args.sample and args.sample < len(pairs):
            random.seed(args.seed)
            pairs = random.sample(pairs, args.sample)
            print(f"Выборка: {len(pairs)} пар (seed={args.seed})")

    print(f"\nВсего пар к показу: {len(pairs)}")

    # 2. скачать всё что нужно
    if not args.skip_download:
        all_urls = []
        for p in pairs:
            all_urls.append(p["before_url"])
            all_urls.append(p["after_url"])
        # дедупликация
        all_urls = list(dict.fromkeys(all_urls))
        print(f"\nСкачиваю фото ({len(all_urls)} URL)...")
        download_all(all_urls, workers=args.workers)
    else:
        print("\nПропускаю скачивание (--skip-download).")

    cached = sum(1 for p in pairs
                 if is_cached(p["before_url"]) and is_cached(p["after_url"]))
    print(f"\nПар, у которых оба фото скачаны: {cached}/{len(pairs)}")

    # 3. GUI
    print("\nОткрываю окно разметки. Прогресс автосохраняется в pair_decisions.csv.")
    root = tk.Tk()
    app = PairFilterApp(root, pairs)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()

    # 4. экспорт по решениям
    decisions = load_decisions()
    if decisions:
        print("\nРаскладываю фото по папкам good_pairs/ и bad_pairs/...")
        export_by_decision(pairs, decisions)
        good = sum(1 for v in decisions.values() if v["decision"] == "good")
        bad = sum(1 for v in decisions.values() if v["decision"] == "bad")
        unsure = sum(1 for v in decisions.values() if v["decision"] == "unsure")
        print(f"  ✓ хороших: {good}")
        print(f"  ✗ плохих: {bad}")
        print(f"  ? не уверен: {unsure}")
    print(f"\nГотово. Решения: {DECISIONS_CSV}")


if __name__ == "__main__":
    main()
