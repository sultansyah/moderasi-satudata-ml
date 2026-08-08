import os
import re
import sys
import shutil
import argparse
import json

import pytesseract
from PIL import Image
from ultralytics import YOLO

from keywords import KEYWORDS_ALL

TESSERACT_CMD_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE, "models", "best.pt")
FALLBACK_MODEL = os.path.join(BASE, "runs", "yolo11n-cls-mod-v4", "weights", "best.pt")

KELAS_VIOLATIVE = ["obat_aborsi"]

# Ambang konfidensi kelas violative untuk YOLO / MobileNetV3 (env VIOL_CONF_THRESHOLD).
# Di bawah ambang, gambar TIDAK langsung DIMODERASI — dilanjut OCR + keyword.
VIOL_CONF_THRESHOLD = float(os.environ.get("VIOL_CONF_THRESHOLD", "0.70"))

# Engine klasifikasi visual: "yolo" (default, lama), "clip" (zero-shot), atau "mobilenetv3".
# Bisa diganti via env MODERASI_VISUAL, argumen --visual, atau runtime (set_visual_engine).
VALID_VISUAL_ENGINES = ("yolo", "clip", "mobilenetv3")
VISUAL_ENGINE = os.environ.get("MODERASI_VISUAL", "yolo").strip().lower()


def set_visual_engine(engine):
    """Ganti engine visual default saat runtime (mis. dari API/UI)."""
    global VISUAL_ENGINE
    engine = engine.strip().lower()
    if engine in VALID_VISUAL_ENGINES:
        VISUAL_ENGINE = engine
        return True
    return False


def visual_engine_available(engine):
    engine = engine.strip().lower()
    if engine == "yolo":
        return True
    if engine == "clip":
        try:
            import transformers  # noqa: F401
            return True
        except Exception:
            return False
    if engine == "mobilenetv3":
        return os.path.isfile(MOBILENETV3_MODEL)
    return False

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_VIOLATIVE_CONCEPTS = [
    ("obat_aborsi", "kemasan obat penggugur kandungan"),
    ("obat_aborsi", "pil atau tablet obat aborsi"),
    ("obat_aborsi", "obat cytotec atau misoprostol"),
    ("obat_aborsi", "produk obat aborsi yang dijual ilegal"),
    ("obat_aborsi", "iklan jual obat aborsi"),
    ("obat_aborsi", "a photo of an abortion pill or its packaging"),
    ("obat_aborsi", "a photo of an advertisement for abortion pills"),
    ("rokok", "a photo of a cigarette or a pack of cigarettes"),
    ("alkohol", "a photo of a bottle of alcohol, beer, or wine"),
    ("narkoba", "a photo of marijuana or illegal drugs"),
    ("dewasa", "a photo of nudity or explicit sexual content"),
    ("dewasa", "a photo of a vulgar or pornographic image"),
    ("kekerasan", "a photo of blood, violence, or a wound"),
    ("senjata", "a photo of a gun, pistol, or sharp weapon"),
    ("judi", "a photo of a slot machine or casino gambling"),
    ("obat_keras", "a photo of medicine blister packs or prescription drugs"),
    ("penipuan", "a photo of a scam advertisement or fake job vacancy poster"),
]
CLIP_SAFE_CONCEPTS = [
    ("dokumen", "dokumen resmi, surat, atau sertifikat"),
    ("dokumen", "a photo of a document, paper, or certificate"),
    ("poster", "poster acara, seminar, atau edukasi"),
    ("poster", "a photo of a poster or banner"),
    ("sertifikat", "sertifikat penghargaan atau pelatihan"),
    ("undangan", "undangan pernikahan atau surat undangan"),
    ("promo", "brosur, banner, atau flyer promosi"),
    ("chat", "screenshot percakapan aplikasi chat atau whatsapp"),
    ("makanan", "a photo of food or a meal"),
    ("orang", "a photo of people"),
    ("hewan", "a photo of an animal, cat, or dog"),
    ("alam", "a photo of a natural landscape or scenery"),
    ("gedung", "a photo of an office building or skyscraper"),
    ("kendaraan", "a photo of a car, motorcycle, or vehicle"),
    ("tanaman", "a photo of green plants or trees"),
    ("sekolah", "a photo of students in a school or classroom"),
    ("grafik", "a photo of a chart, graph, or data visualization"),
    ("produk", "a photo of a product, cosmetics, or household goods"),
    ("game", "a screenshot of a video game"),
    ("anime", "a friendly anime or cartoon illustration"),
    ("fashion", "a photo of clothes, bags, or shoes"),
    ("resep", "a photo of a recipe or cooked dish"),
    ("rumah", "a photo of a house or residential building"),
    ("normal", "a photo of a normal everyday scene"),
]

_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _clip_model.eval()
    return _clip_model, _clip_processor


def _clip_classify(image_path):
    import torch
    model, proc = _load_clip()
    image = Image.open(image_path).convert("RGB")
    vio_labels = [l for l, _ in CLIP_VIOLATIVE_CONCEPTS]
    vio_texts = [t for _, t in CLIP_VIOLATIVE_CONCEPTS]
    safe_texts = [t for _, t in CLIP_SAFE_CONCEPTS]
    texts = vio_texts + safe_texts
    inputs = proc(text=texts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image[0]
        probs = logits.softmax(dim=-1).tolist()
    n_vio = len(vio_texts)
    vio_scores = probs[:n_vio]
    safe_scores = probs[n_vio:]
    vio_sum = sum(vio_scores)
    safe_sum = sum(safe_scores)
    floor = float(os.environ.get("CLIP_VIOL_THRESHOLD", "0.30"))
    margin = float(os.environ.get("CLIP_VIOL_MARGIN", "2.0"))
    best_vio = max(vio_scores)
    best_safe = max(safe_scores)
    violative = vio_sum >= floor and vio_sum >= safe_sum * margin
    if violative:
        cls_name = vio_labels[vio_scores.index(best_vio)]
    else:
        cls_name = "aman"
    conf = max(best_vio, best_safe)
    return cls_name, round(float(conf), 4), violative


MOBILENETV3_MODEL = os.path.join(BASE, "models", "mobilenetv3_best.pt")
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

_mobilenetv3 = None


def _load_mobilenetv3():
    global _mobilenetv3
    if _mobilenetv3 is None:
        if not os.path.isfile(MOBILENETV3_MODEL):
            raise FileNotFoundError(f"Model MobileNetV3 tidak ditemukan: {MOBILENETV3_MODEL}")
        import torch
        from torchvision import models, transforms
        ckpt = torch.load(MOBILENETV3_MODEL, map_location="cpu")
        class_names = list(ckpt["class_names"])
        model = models.mobilenet_v3_large()
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(class_names))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        img_size = int(ckpt.get("img_size", 224))
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
        ])
        _mobilenetv3 = (model, class_names, tf)
    return _mobilenetv3


def _mobilenetv3_classify(image_path):
    import torch
    model, class_names, tf = _load_mobilenetv3()
    img = tf(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(img), dim=1)[0]
    idx = int(probs.argmax())
    cls_name = class_names[idx]
    conf = round(float(probs[idx]), 4)
    violative = cls_name in KELAS_VIOLATIVE and conf >= VIOL_CONF_THRESHOLD
    return cls_name, conf, violative


def _visual_classify(model, image_path, engine=None):
    """Klasifikasi visual. Mengembalikan (cls_name, conf, violative, engine)."""
    engine = (engine or VISUAL_ENGINE).strip().lower()
    if engine == "clip":
        try:
            return (*_clip_classify(image_path), "clip")
        except Exception as e:
            print(f"[WARN] CLIP gagal ({e}); fallback ke YOLO.")
    if engine == "mobilenetv3":
        try:
            return (*_mobilenetv3_classify(image_path), "mobilenetv3")
        except Exception as e:
            print(f"[WARN] MobileNetV3 gagal ({e}); fallback ke YOLO.")
    preds = model.predict(image_path, verbose=False)
    if preds:
        p = preds[0]
        if p.probs is not None:
            cls_name = model.names[int(p.probs.top1)]
            conf = round(float(p.probs.top1conf), 4)
            violative = cls_name in KELAS_VIOLATIVE and conf >= VIOL_CONF_THRESHOLD
            return cls_name, conf, violative, "yolo"
    return None, None, False, "yolo"


def setup_tesseract():
    if os.path.isfile(TESSERACT_CMD_WIN):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_WIN
        return
    exe = shutil.which("tesseract")
    pytesseract.pytesseract.tesseract_cmd = exe or "tesseract"

def load_model(path=None):
    model_path = path or os.environ.get("MODERASI_MODEL") or DEFAULT_MODEL
    if not os.path.isfile(model_path) and os.path.isfile(FALLBACK_MODEL):
        model_path = FALLBACK_MODEL
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        sys.exit(1)
    setup_tesseract()
    return YOLO(model_path)


MAX_OCR_SIZE = 1600
OCR_CONFIG = "--oem 1 --psm 3"


def _prep_ocr_image(image_path):
    img = Image.open(image_path).convert("L")
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_OCR_SIZE:
        scale = MAX_OCR_SIZE / longest
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return img


def ocr_text(image_path, lang="ind+eng"):
    try:
        img = _prep_ocr_image(image_path)
        txt = pytesseract.image_to_string(img, lang=lang, config=OCR_CONFIG)
        return txt.strip()
    except Exception as e:
        print(f"[WARN] OCR gagal: {e}")
        return ""


def normalize(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def keyword_hits(text_normalized):
    hits = []
    for kw in KEYWORDS_ALL:
        if kw in text_normalized:
            hits.append(kw)
    return hits


def moderasi_satu_gambar(model, image_path, lang="ind+eng", visual=None):
    filename = os.path.basename(image_path)
    result = {
        "file": image_path,
        "filename": filename,
        "visual_engine": None,
        "yolo_class": None,
        "yolo_conf": None,
        "yolo_violative": False,
        "ocr_text": "",
        "keyword_hits": [],
        "keputusan": "LOLOS",
        "alasan": [],
    }

    # 1) Visual classification (CLIP zero-shot, MobileNetV3, atau YOLO)
    cls_name, conf, violative, engine = _visual_classify(model, image_path, engine=visual)
    result["visual_engine"] = engine
    result["yolo_class"] = cls_name
    result["yolo_conf"] = conf
    if violative:
        result["yolo_violative"] = True
        result["alasan"].append(f"{engine.upper()} deteksi kelas violative: {cls_name} (conf {conf:.2f})")

    # 2) Tesseract OCR — dilewati jika visual sudah violative (keputusan sudah DIMODERASI)
    if not result["yolo_violative"]:
        raw = ocr_text(image_path, lang=lang)
        result["ocr_text"] = raw[:500]
        if raw:
            norm = normalize(raw)
            hits = keyword_hits(norm)
            if hits:
                result["keyword_hits"] = hits[:20]
                result["alasan"].append(f"OCR match keyword: {', '.join(hits[:10])}")

        # 3) Filename check (fallback kalau OCR kosong)
        if not result["keyword_hits"]:
            hits = keyword_hits(normalize(filename))
            if hits:
                result["keyword_hits"] = hits[:20]
                result["alasan"].append(f"Filename match keyword: {', '.join(hits[:10])}")
    else:
        result["alasan"].append("OCR dilewati (visual sudah violative)")

    # Keputusan akhir
    if result["yolo_violative"] or result["keyword_hits"]:
        result["keputusan"] = "DIMODERASI"
    return result


def main():
    ap = argparse.ArgumentParser(description="Sistem Moderasi Gambar Otomatis")
    ap.add_argument("input", nargs="+", help="Path gambar atau folder")
    ap.add_argument("--model", default=None, help="Path model YOLO (default: best.pt dari training)")
    ap.add_argument("--visual", default=None, choices=["yolo", "clip", "mobilenetv3"], help="Engine visual: yolo | clip | mobilenetv3 (default: env MODERASI_VISUAL atau yolo)")
    ap.add_argument("--lang", default="ind+eng", help="Bahasa OCR Tesseract (default: ind+eng)")
    ap.add_argument("--json", action="store_true", help="Output JSON ke konsol")
    args = ap.parse_args()

    if args.visual:
        global VISUAL_ENGINE
        VISUAL_ENGINE = args.visual

    model = load_model(args.model)

    files = []
    for inp in args.input:
        if os.path.isfile(inp):
            files.append(inp)
        elif os.path.isdir(inp):
            for root, dirs, fs in os.walk(inp):
                for f in fs:
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")):
                        files.append(os.path.join(root, f))
        else:
            print(f"[WARN] Path tidak valid: {inp}")

    if not files:
        print("[ERROR] Tidak ada gambar ditemukan.")
        sys.exit(1)

    print(f"\n=== MODERASI {len(files)} GAMBAR ===\n")
    summary = {"LOLOS": 0, "DIMODERASI": 0}
    all_results = []
    for f in files:
        r = moderasi_satu_gambar(model, f, lang=args.lang)
        all_results.append(r)
        summary[r["keputusan"]] += 1
        status = "[DIMODERASI]" if r["keputusan"] == "DIMODERASI" else "[LOLOS]"
        print(f"{status} {os.path.basename(f)}")
        print(f"   {r['visual_engine'].upper()} : {r['yolo_class']} (conf {r['yolo_conf']})")
        if r["keyword_hits"]:
            print(f"   KEY  : {', '.join(r['keyword_hits'][:8])}")
        if r["alasan"]:
            print(f"   ALASAN: {'; '.join(r['alasan'][:3])}")

    print(f"\n=== RINGKASAN: LOLOS={summary['LOLOS']} DIMODERASI={summary['DIMODERASI']} ===")
    if args.json:
        print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
