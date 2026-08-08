import os
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, "dataset")

assert os.path.isdir(DATASET), f"Dataset tidak ditemukan: {DATASET}"

MODEL = "yolo11n-cls.pt"
EPOCHS = 40
IMGSZ = 224
BATCH = 32
PATIENCE = 12
WORKERS = 0
CACHE = "ram"
NAME = "yolo11n-cls-mod-v5"

# Augmentasi online (diterapkan ultralytics tiap epoch, hanya data train)
AUGMENT = True
DEGREES = 5          # rotasi acak derajat
TRANSLATE = 0.1      # geser acak
SCALE = 0.2          # skala acak
FLIPUD = 0.05        # flip vertikal
MIXUP = 0.1          # campur 2 gambar
# hsv_h/s/v, erasing (cutout), auto_augment randaugment memakai bawaan ultralytics

print("=== Training YOLO11-cls (3 kelas: obat_aborsi / normal / dokumen) ===")
print(f"  dataset : {DATASET}")
print(f"  model   : {MODEL}")
print(f"  epochs  : {EPOCHS}  batch: {BATCH}  imgsz: {IMGSZ}")
print(f"  augment : {AUGMENT} (degrees={DEGREES}, translate={TRANSLATE}, scale={SCALE},"
      f" flipud={FLIPUD}, mixup={MIXUP})")

model = YOLO(MODEL)

results = model.train(
    data=DATASET,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=PATIENCE,
    workers=WORKERS,
    device="cpu",
    project=os.path.join(BASE, "runs"),
    name=NAME,
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

metrics = model.val(project=os.path.join(BASE, "runs"), name=f"{NAME}-val")

print("\n=== TRAINING SELESAI ===")
print(f"  best model: {os.path.join(BASE, 'runs', NAME, 'weights', 'best.pt')}")
print(f"  top1 acc  : {metrics.top1:.4f}")
print(f"  top5 acc  : {metrics.top5:.4f}")
print(f"  classes   : {model.names}")
