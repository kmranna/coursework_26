#!/usr/bin/env python3
"""Визуализация QA пар v3 — без чёрных полос."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF


class RandomGamma(object):
    def __init__(self, gamma_range=(0.6, 1.6), p=0.7):
        self.gamma_range = gamma_range; self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = TF.adjust_gamma(img, gamma=random.uniform(*self.gamma_range))
        return img


class RandomColorTemperature(object):
    def __init__(self, strength=0.15, p=0.5):
        self.strength = strength; self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            arr = np.array(img).astype(np.float32) / 255.0
            shift = random.uniform(-self.strength, self.strength)
            arr[..., 0] = np.clip(arr[..., 0] + shift, 0, 1)
            arr[..., 2] = np.clip(arr[..., 2] - shift, 0, 1)
            img = Image.fromarray((arr * 255).astype(np.uint8))
        return img


class SafeGeometricAug(object):
    def __init__(self, image_size, rotation_deg=12, perspective_dist=0.25,
                 affine_translate=0.08, affine_shear=8, p_perspective=0.7, p_hflip=0.3):
        self.image_size = image_size
        self.pad_size = image_size // 4
        self.rotation_deg = rotation_deg
        self.perspective_dist = perspective_dist
        self.affine_translate = affine_translate
        self.affine_shear = affine_shear
        self.p_perspective = p_perspective
        self.p_hflip = p_hflip

    def __call__(self, img):
        target = self.image_size + 2 * self.pad_size
        img = TF.resize(img, [target, target])
        if random.random() < self.p_hflip:
            img = TF.hflip(img)
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        img = TF.rotate(img, angle, fill=0)
        translate_x = random.uniform(-self.affine_translate, self.affine_translate) * target
        translate_y = random.uniform(-self.affine_translate, self.affine_translate) * target
        shear_x = random.uniform(-self.affine_shear, self.affine_shear)
        img = TF.affine(img, angle=0, translate=[int(translate_x), int(translate_y)],
                         scale=1.0, shear=[shear_x, 0], fill=0)
        if random.random() < self.p_perspective:
            startpoints = [[0, 0], [target, 0], [target, target], [0, target]]
            d = int(self.perspective_dist * target)
            endpoints = [
                [random.randint(0, d), random.randint(0, d)],
                [random.randint(target - d, target), random.randint(0, d)],
                [random.randint(target - d, target), random.randint(target - d, target)],
                [random.randint(0, d), random.randint(target - d, target)],
            ]
            img = TF.perspective(img, startpoints, endpoints, fill=0)
        img = TF.center_crop(img, [self.image_size, self.image_size])
        return img


def base_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
        RandomGamma(gamma_range=(0.75, 1.35), p=0.5),
        transforms.ToTensor(),
    ])


def synthetic_after_transform(image_size):
    return transforms.Compose([
        SafeGeometricAug(image_size, rotation_deg=12, perspective_dist=0.25,
                         affine_translate=0.08, affine_shear=8, p_perspective=0.7, p_hflip=0.3),
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
        RandomGamma(gamma_range=(0.5, 1.7), p=0.8),
        RandomColorTemperature(strength=0.18, p=0.6),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.4),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 1.5)),
        transforms.ToTensor(),
    ])


def tensor_to_pil(t):
    arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(arr)


def make_side_by_side(img_before, img_after, label, synthetic, image_size=224):
    gap = 8
    title_h = 32
    canvas = Image.new('RGB', (image_size * 2 + gap, image_size + title_h), 'white')
    canvas.paste(img_before, (0, title_h))
    canvas.paste(img_after, (image_size + gap, title_h))
    draw = ImageDraw.Draw(canvas)
    if label == 1:
        title = "POSITIVE (label=1, исправлено)"
        color = (0, 130, 0)
    else:
        title = f"NEGATIVE (label=0, synthetic={synthetic}, 'не исправлено')"
        color = (180, 0, 0)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    draw.text((6, 6), title, fill=color, font=font)
    draw.text((6, 22), "BEFORE (with aug)             AFTER (with aug)",
              fill=(80, 80, 80), font=font)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="vis_qa")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.csv, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    pos = [r for r in rows if r['label'] == '1']
    neg = [r for r in rows if r['label'] == '0']
    print(f"Всего пар: positive={len(pos)}, negative (synthetic)={len(neg)}")

    random.seed(args.seed)
    n_per_class = args.n // 2
    sample_pos = random.sample(pos, min(n_per_class, len(pos)))
    sample_neg = random.sample(neg, min(n_per_class, len(neg)))

    base_tf = base_transform(args.image_size)
    syn_tf = synthetic_after_transform(args.image_size)

    saved = 0
    for kind, sample in [('positive', sample_pos), ('negative', sample_neg)]:
        for i, r in enumerate(sample):
            try:
                img_before = Image.open(r['path_before']).convert('RGB')
                img_after = Image.open(r['path_after']).convert('RGB')
            except Exception as e:
                print(f"Skip: {e}"); continue
            is_synthetic = int(r.get('synthetic', 0)) == 1
            label = int(r['label'])
            img_before_t = base_tf(img_before)
            img_after_t = syn_tf(img_after) if is_synthetic else base_tf(img_after)
            canvas = make_side_by_side(tensor_to_pil(img_before_t), tensor_to_pil(img_after_t),
                                        label, int(r.get('synthetic', 0)), args.image_size)
            canvas.save(out_dir / f"{kind}_{i:02d}.png")
            saved += 1
    print(f"\nСохранено {saved} визуализаций в {out_dir}/")


if __name__ == "__main__":
    main()
