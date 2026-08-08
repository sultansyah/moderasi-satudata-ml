import argparse
import os
import random
import shutil

from PIL import Image, ImageEnhance

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DST = os.path.join(BASE, "dataset")
VAL_RATIO = 0.2
SEED = 42
DEFAULT_AUG = 2

DEFAULT_SOURCES = {
    "obat_aborsi": r"C:\Users\sulta\Desktop\dataset_obat_aborsi_google",
    "normal": r"C:\Users\sulta\Desktop\dataset_normal",
    "dokumen": r"C:\Users\sulta\Desktop\dataset_dokumen",
}

random.seed(SEED)


def augment_variant(im):
    w, h = im.size
    # zoom/crop kecil (10-15% masuk frame)
    if random.random() < 0.5:
        s = random.uniform(0.85, 1.0)
        nw, nh = max(16, int(w * s)), max(16, int(h * s))
        x0 = random.randint(0, w - nw)
        y0 = random.randint(0, h - nh)
        im = im.crop((x0, y0, x0 + nw, y0 + nh)).resize((w, h), Image.BILINEAR)
    # rotasi kecil
    if random.random() < 0.7:
        im = im.rotate(random.uniform(-12, 12), resample=Image.BILINEAR, fillcolor=255)
    # flip horizontal
    if random.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    # kecerahan & kontras
    im = ImageEnhance.Brightness(im).enhance(random.uniform(0.85, 1.15))
    im = ImageEnhance.Contrast(im).enhance(random.uniform(0.85, 1.15))
    return im


def build(sources, dst, aug_per_image):
    for split in ("train", "val"):
        split_dir = os.path.join(dst, split)
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)

    summary = {}
    for cls, src in sources.items():
        if not os.path.isdir(src):
            print(f"[WARN] source tidak ada: {src}")
            continue
        files = []
        for root, dirs, fs in os.walk(src):
            for f in fs:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")):
                    files.append(os.path.join(root, f))
        random.shuffle(files)
        n_val = max(1, int(len(files) * VAL_RATIO))
        val_files, train_files = files[:n_val], files[n_val:]

        d_train = os.path.join(dst, "train", cls)
        d_val = os.path.join(dst, "val", cls)
        os.makedirs(d_train, exist_ok=True)
        os.makedirs(d_val, exist_ok=True)

        for i, f in enumerate(train_files):
            shutil.copy2(f, os.path.join(d_train, f"{i:04d}{os.path.splitext(f)[1].lower()}"))
        for i, f in enumerate(val_files):
            shutil.copy2(f, os.path.join(d_val, f"{i:04d}{os.path.splitext(f)[1].lower()}"))

        n_aug = 0
        if aug_per_image > 0:
            for i, f in enumerate(train_files):
                try:
                    im = Image.open(f).convert("RGB")
                except Exception:
                    continue
                base = os.path.join(d_train, f"{i:04d}")
                for n in range(aug_per_image):
                    try:
                        augment_variant(im).save(f"{base}_a{n:02d}.jpg", "JPEG", quality=90)
                        n_aug += 1
                    except Exception:
                        pass

        summary[cls] = (len(train_files) + n_aug, len(val_files), len(files))
        print(f"  {cls}: train={len(train_files)}+{n_aug}aug val={len(val_files)} total={len(files)}")

    tot_train = sum(v[0] for v in summary.values())
    tot_val = sum(v[1] for v in summary.values())
    print(f"\nTOTAL train={tot_train} val={tot_val} kelas={list(summary.keys())} (augmentasi: {aug_per_image}x per gambar train)")


def parse_sources(s):
    out = {}
    for item in s.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser(description="Bangun dataset train/val + augmentasi offline")
    ap.add_argument("--dst", default=DEFAULT_DST, help="folder output dataset")
    ap.add_argument("--sources", default=None,
                    help="override sumber: cls=path;cls=path (default = folder Desktop)")
    ap.add_argument("--aug", type=int, default=DEFAULT_AUG,
                    help="varian augmentasi per gambar train (0 = nonaktif)")
    args = ap.parse_args()
    sources = parse_sources(args.sources) if args.sources else DEFAULT_SOURCES
    build(sources, args.dst, args.aug)


if __name__ == "__main__":
    main()
