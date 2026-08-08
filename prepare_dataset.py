import os
import random
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(BASE, "dataset")
VAL_RATIO = 0.2
SEED = 42

SOURCES = {
    "obat_aborsi": r"C:\Users\sulta\Desktop\dataset_obat_aborsi_google",
    "normal": r"C:\Users\sulta\Desktop\dataset_normal",
}

random.seed(SEED)

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

    summary[cls] = (len(train_files), len(val_files), len(files))
    print(f"  {cls}: train={len(train_files)} val={len(val_files)} total={len(files)}")

tot_train = sum(v[0] for v in summary.values())
tot_val = sum(v[1] for v in summary.values())
print(f"\nTOTAL train={tot_train} val={tot_val} kelas={list(summary.keys())}")
