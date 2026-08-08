import os
import random
import shutil

from PIL import Image, ImageEnhance

BASE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(BASE, "dataset")
VAL_RATIO = 0.2
SEED = 42

# Augmentasi offline: jumlah varian tambahan per gambar TRAIN (0 = nonaktif).
# Hanya train yang di-augment (val tetap asli agar evaluasi jujur).
AUG_PER_IMAGE = 2

SOURCES = {
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


def build():
    for split in ("train", "val"):
        split_dir = os.path.join(DST, split)
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)

    summary = {}
    for cls, src in SOURCES.items():
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

        d_train = os.path.join(DST, "train", cls)
        d_val = os.path.join(DST, "val", cls)
        os.makedirs(d_train, exist_ok=True)
        os.makedirs(d_val, exist_ok=True)

        for i, f in enumerate(train_files):
            shutil.copy2(f, os.path.join(d_train, f"{i:04d}{os.path.splitext(f)[1].lower()}"))
        for i, f in enumerate(val_files):
            shutil.copy2(f, os.path.join(d_val, f"{i:04d}{os.path.splitext(f)[1].lower()}"))

        n_aug = 0
        if AUG_PER_IMAGE > 0:
            for i, f in enumerate(train_files):
                try:
                    im = Image.open(f).convert("RGB")
                except Exception:
                    continue
                base = os.path.join(d_train, f"{i:04d}")
                for n in range(AUG_PER_IMAGE):
                    try:
                        augment_variant(im).save(f"{base}_a{n:02d}.jpg", "JPEG", quality=90)
                        n_aug += 1
                    except Exception:
                        pass

        summary[cls] = (len(train_files) + n_aug, len(val_files), len(files))
        print(f"  {cls}: train={len(train_files)}+{n_aug}aug val={len(val_files)} total={len(files)}")

    tot_train = sum(v[0] for v in summary.values())
    tot_val = sum(v[1] for v in summary.values())
    print(f"\nTOTAL train={tot_train} val={tot_val} kelas={list(summary.keys())} (augmentasi: {AUG_PER_IMAGE}x per gambar train)")


if __name__ == "__main__":
    build()
