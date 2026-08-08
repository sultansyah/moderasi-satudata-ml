import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Large_Weights

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE, "dataset")
DEFAULT_OUT = os.path.join(BASE, "models", "mobilenetv3_best.pt")
IMG_SIZE = 224
NUM_WORKERS = 0

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def build_model(num_classes, pretrained=True):
    """MobileNetV3-Large + head 3 kelas (transfer learning dari ImageNet)."""
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, device, class_names):
    model.eval()
    correct = 0
    total = 0
    per_cls = {c: [0, 0] for c in class_names}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for p, t in zip(pred.tolist(), y.tolist()):
                per_cls[class_names[t]][1] += 1
                if p == t:
                    per_cls[class_names[t]][0] += 1
    acc = correct / total if total else 0.0
    per = {c: (v[0] / v[1] if v[1] else 0.0) for c, v in per_cls.items()}
    return acc, per


def main():
    ap = argparse.ArgumentParser(description="Training MobileNetV3-Large (obat_aborsi / normal / dokumen)")
    ap.add_argument("--data", default=DEFAULT_DATA, help="folder dataset (berisi train/ val/)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="path checkpoint hasil training")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--from-scratch", action="store_true", help="tanpa pretrained ImageNet")
    ap.add_argument("--device", default="auto", help="cpu / cuda:0 / auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    assert os.path.isdir(args.data), f"Dataset tidak ditemukan: {args.data}"

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(args.data, "val"), transform=val_tf)
    class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=NUM_WORKERS)

    print("=== Training MobileNetV3-Large (3 kelas: obat_aborsi / normal / dokumen) ===")
    print(f"  data      : {args.data}  ({len(train_ds)} train / {len(val_ds)} val)")
    print(f"  classes   : {class_names}")
    print(f"  epochs    : {args.epochs}  batch: {args.batch}  lr: {args.lr}  device: {device}")
    print(f"  pretrained: {not args.from_scratch}   out: {args.out}")

    model = build_model(len(class_names), pretrained=not args.from_scratch).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_acc = 0.0
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()

        val_acc, per = evaluate(model, val_loader, device, class_names)
        print(f"[{epoch + 1}/{args.epochs}] loss={train_loss / total:.4f} "
              f"train_acc={correct / total:.4f} val_acc={val_acc:.4f} ({time.time() - t0:.0f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "img_size": IMG_SIZE,
                "val_acc": val_acc,
                "num_classes": len(class_names),
            }, args.out)
            print(f"    * best tersimpan -> {args.out} (val_acc={val_acc:.4f})")

    print("\n=== TRAINING SELESAI ===")
    print(f"  waktu     : {time.time() - t_start:.0f}s")
    print(f"  best model: {args.out}")
    print(f"  val_acc   : {best_acc:.4f}")
    print(f"  per kelas : {per}")


if __name__ == "__main__":
    main()
