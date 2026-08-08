import argparse
import os

import torch
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE, "dataset")

# Augmentasi online (diterapkan ultralytics tiap epoch, hanya data train)
AUGMENT = True
DEGREES = 5          # rotasi acak derajat
TRANSLATE = 0.1      # geser acak
SCALE = 0.2          # skala acak
FLIPUD = 0.05        # flip vertikal
MIXUP = 0.1          # campur 2 gambar
# hsv_h/s/v, erasing (cutout), auto_augment randaugment memakai bawaan ultralytics


def main():
    ap = argparse.ArgumentParser(description="Training YOLO11-cls (obat_aborsi / normal / dokumen)")
    ap.add_argument("--data", default=DEFAULT_DATA, help="folder dataset (berisi train/ val/)")
    ap.add_argument("--model", default="yolo11n-cls.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto", help="cpu / cuda:0 / auto")
    ap.add_argument("--name", default="yolo11n-cls-mod-v5")
    ap.add_argument("--project", default=os.path.join(BASE, "runs"))
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    assert os.path.isdir(args.data), f"Dataset tidak ditemukan: {args.data}"
    print("=== Training YOLO11-cls (3 kelas: obat_aborsi / normal / dokumen) ===")
    print(f"  data    : {args.data}")
    print(f"  model   : {args.model}")
    print(f"  epochs  : {args.epochs}  batch: {args.batch}  imgsz: {args.imgsz}  device: {device}")
    print(f"  augment : {AUGMENT} (degrees={DEGREES}, translate={TRANSLATE}, scale={SCALE},"
          f" flipud={FLIPUD}, mixup={MIXUP})")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        pretrained=True,
        seed=42,
        verbose=True,
        augment=AUGMENT,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        flipud=FLIPUD,
        mixup=MIXUP,
    )

    metrics = model.val(project=args.project, name=f"{args.name}-val")

    print("\n=== TRAINING SELESAI ===")
    print(f"  best model: {os.path.join(args.project, args.name, 'weights', 'best.pt')}")
    print(f"  top1 acc  : {metrics.top1:.4f}")
    print(f"  top5 acc  : {metrics.top5:.4f}")
    print(f"  classes   : {model.names}")


if __name__ == "__main__":
    main()
