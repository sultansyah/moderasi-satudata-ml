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
FALLBACK_MODEL = os.path.join(BASE, "runs", "yolo11n-cls-mod-v3", "weights", "best.pt")

KELAS_VIOLATIVE = ["obat_aborsi"]

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


def moderasi_satu_gambar(model, image_path, lang="ind+eng"):
    filename = os.path.basename(image_path)
    result = {
        "file": image_path,
        "filename": filename,
        "yolo_class": None,
        "yolo_conf": None,
        "yolo_violative": False,
        "ocr_text": "",
        "keyword_hits": [],
        "keputusan": "LOLOS",
        "alasan": [],
    }

    # 1) YOLO classification
    preds = model.predict(image_path, verbose=False)
    if preds:
        p = preds[0]
        if p.probs is not None:
            cls_id = int(p.probs.top1)
            conf = float(p.probs.top1conf)
            cls_name = model.names[cls_id]
            result["yolo_class"] = cls_name
            result["yolo_conf"] = round(conf, 4)
            if cls_name in KELAS_VIOLATIVE:
                result["yolo_violative"] = True
                result["alasan"].append(f"YOLO deteksi kelas violative: {cls_name} (conf {conf:.2f})")

    # 2) Tesseract OCR
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

    # Keputusan akhir
    if result["yolo_violative"] or result["keyword_hits"]:
        result["keputusan"] = "DIMODERASI"
    return result


def main():
    ap = argparse.ArgumentParser(description="Sistem Moderasi Gambar Otomatis")
    ap.add_argument("input", nargs="+", help="Path gambar atau folder")
    ap.add_argument("--model", default=None, help="Path model YOLO (default: best.pt dari training)")
    ap.add_argument("--lang", default="ind+eng", help="Bahasa OCR Tesseract (default: ind+eng)")
    ap.add_argument("--json", action="store_true", help="Output JSON ke konsol")
    args = ap.parse_args()

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
        print(f"   YOLO : {r['yolo_class']} (conf {r['yolo_conf']})")
        if r["keyword_hits"]:
            print(f"   KEY  : {', '.join(r['keyword_hits'][:8])}")
        if r["alasan"]:
            print(f"   ALASAN: {'; '.join(r['alasan'][:3])}")

    print(f"\n=== RINGKASAN: LOLOS={summary['LOLOS']} DIMODERASI={summary['DIMODERASI']} ===")
    if args.json:
        print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
