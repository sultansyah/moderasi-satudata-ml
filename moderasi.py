import os
import re
import sys
import shutil
import argparse
import json
import unicodedata

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

# Engine klasifikasi visual: "yolo" (default, lama), "clip" (zero-shot),
# "mobilenetv3", atau "smolvlm" (VLM SmolVLM2-256M).
# Bisa diganti via env MODERASI_VISUAL, argumen --visual, atau runtime (set_visual_engine).
VALID_VISUAL_ENGINES = ("yolo", "clip", "mobilenetv3", "smolvlm")
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
    if engine in ("clip", "smolvlm"):
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
    ("obat_aborsi", "iklan yang menjual obat aborsi dengan harga atau kontak pemesanan"),
    ("obat_aborsi", "promosi jual obat cytotec atau misoprostol"),
    ("obat_aborsi", "poster jual obat penggugur kandungan"),
    ("obat_aborsi", "an advertisement selling abortion pills with price or contact number"),
    ("obat_aborsi", "a promotional poster for buying abortion pills"),
    ("boraks", "iklan yang menjual boraks atau formalin untuk makanan"),
    ("boraks", "promosi jual bahan kimia boraks dengan harga atau kontak pemesanan"),
    ("boraks", "an advertisement selling borax or formalin as a food additive"),
    ("judi", "iklan judi online yang mengajak daftar atau deposit"),
    ("judi", "banner promosi slot gacor, maxwin, bonus, atau situs taruhan"),
    ("judi", "a gambling advertisement asking users to register or deposit money"),
    ("judi", "a promotional online casino or slot betting banner"),
    ("rokok", "a promotional advertisement selling cigarettes"),
    ("alkohol", "a promotional advertisement selling alcohol, beer, or wine"),
    ("narkoba", "an advertisement selling marijuana or illegal drugs"),
    ("dewasa", "a promotional pornographic or explicit sexual image"),
    ("kekerasan", "a graphic image of blood, violence, or a wound"),
    ("senjata", "an advertisement selling a gun, pistol, or sharp weapon"),
    ("obat_keras", "an advertisement selling prescription drugs without context"),
    ("penipuan", "a scam advertisement or fake job vacancy poster"),
]
CLIP_SAFE_CONCEPTS = [
    ("dokumen", "dokumen resmi, surat, atau sertifikat"),
    ("dokumen", "a photo of a document, paper, or certificate"),
    ("poster", "poster acara, seminar, atau edukasi"),
    ("poster", "a photo of a poster or banner"),
    ("berita", "ilustrasi berita tentang spam, peretasan, atau konten ilegal"),
    ("berita", "a news illustration about cyber spam, hacked websites, or illegal content"),
    ("edukasi", "poster edukasi pemerintah yang memperingatkan bahaya boraks, obat ilegal, atau judi online"),
    ("edukasi", "a public warning poster about the dangers of borax, illegal medicine, or online gambling"),
    ("edukasi", "a government campaign poster against illegal drugs, borax, or gambling"),
    ("edukasi", "an anti gambling public service announcement telling people to stop gambling"),
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


SMOLVLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Instruct"
SMOLVLM_PROMPT = "Classify this image: ILLEGAL_AD, PUBLIC_INFO, or NORMAL."

_smolvlm_model = None
_smolvlm_processor = None


def _load_smolvlm():
    global _smolvlm_model, _smolvlm_processor
    if _smolvlm_model is None:
        import torch
        from transformers import AutoProcessor
        from transformers import AutoModelForImageTextToText as _AutoVLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _smolvlm_processor = AutoProcessor.from_pretrained(
            SMOLVLM_MODEL_ID, size={"longest_edge": int(os.environ.get("SMOLVLM_MAX_SIZE", "1024"))}
        )
        _smolvlm_model = _AutoVLM.from_pretrained(
            SMOLVLM_MODEL_ID, dtype=dtype
        ).to(device)
        _smolvlm_model.eval()
    return _smolvlm_model, _smolvlm_processor


def _smolvlm_classify(image_path):
    import torch
    model, proc = _load_smolvlm()
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": SMOLVLM_PROMPT},
        ]},
    ]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=prompt, images=[image])
    for k in ("input_ids", "attention_mask"):
        inputs[k] = torch.tensor(inputs[k])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                   pad_token_id=proc.tokenizer.eos_token_id)
    output_ids = generated[0][inputs["input_ids"].shape[1]:]
    answer = proc.batch_decode(output_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
    upper = answer.upper()
    cats = []
    if "ILLEGAL_AD" in upper:
        cats.append("ILLEGAL_AD")
    if "PUBLIC_INFO" in upper:
        cats.append("PUBLIC_INFO")
    if "NORMAL" in upper:
        cats.append("NORMAL")
    if len(cats) == 1:
        verdict = cats[0]
    elif not cats:
        if "ILLEGAL" in upper:
            verdict = "ILLEGAL_AD"
        elif any(m in upper for m in ("PUBLIC", "NEWS", "WARNING", "EDUCAT", "CAMPAIGN")):
            verdict = "PUBLIC_INFO"
        else:
            verdict = "NORMAL"
    else:
        verdict = "NORMAL"
    violative = verdict == "ILLEGAL_AD"
    return verdict, None, violative, answer


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
    """Klasifikasi visual. Mengembalikan (cls_name, conf, violative, engine, detail)."""
    engine = (engine or VISUAL_ENGINE).strip().lower()
    if engine == "clip":
        try:
            return (*_clip_classify(image_path), "clip", None)
        except Exception as e:
            print(f"[WARN] CLIP gagal ({e}); fallback ke YOLO.")
    if engine == "smolvlm":
        try:
            cls_name, conf, violative, answer = _smolvlm_classify(image_path)
            return cls_name, conf, violative, "smolvlm", answer
        except Exception as e:
            print(f"[WARN] SmolVLM gagal ({e}); fallback ke YOLO.")
    if engine == "mobilenetv3":
        try:
            return (*_mobilenetv3_classify(image_path), "mobilenetv3", None)
        except Exception as e:
            print(f"[WARN] MobileNetV3 gagal ({e}); fallback ke YOLO.")
    preds = model.predict(image_path, verbose=False)
    if preds:
        p = preds[0]
        if p.probs is not None:
            cls_name = model.names[int(p.probs.top1)]
            conf = round(float(p.probs.top1conf), 4)
            violative = cls_name in KELAS_VIOLATIVE and conf >= VIOL_CONF_THRESHOLD
            return cls_name, conf, violative, "yolo", None
    return None, None, False, "yolo", None


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


def _strip_vietnamese_diacritics(s):
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("đ", "d").replace("Đ", "d")
    s = s.replace("ư", "u").replace("Ư", "u")
    s = s.replace("ơ", "o").replace("Ơ", "o")
    return "".join(c for c in s if not unicodedata.combining(c))


def normalize(s):
    s = s.lower()
    s = _strip_vietnamese_diacritics(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def keyword_hits(text_normalized):
    hits = []
    words = set(text_normalized.split())
    for kw in KEYWORDS_ALL:
        if len(kw) <= 3:
            if kw in words:
                hits.append(kw)
        else:
            if kw in text_normalized:
                hits.append(kw)
    return hits


PUBLIC_INTEREST_KEYWORDS = [
    "berita",
    "berita nasional",
    "skandal",
    "siber",
    "cyber",
    "spam",
    "pemerintah",
    "kominfo",
    "investigasi",
    "keamanan data",
    "dipertanyakan",
    "publik resah",
    "desakan",
    "peringatan",
    "waspada",
    "hati hati",
    "bahaya",
    "dampak",
    "edukasi",
    "sosialisasi",
    "kampanye",
    "pencegahan",
    "cegah",
    "larangan",
    "anti judi",
    "jangan judi",
    "jangan biarkan",
    "berhenti sekarang",
    "konseling",
    "adiksi",
    "lapor",
    "imbauan",
    "edaran",
    "informasi",
]

TRANSACTION_KEYWORDS = [
    "jual",
    "beli",
    "order",
    "pesan",
    "pemesanan",
    "stok",
    "ready",
    "harga",
    "promo",
    "diskon",
    "wa",
    "whatsapp",
    "cod",
    "kontak",
    "hubungi",
    "nomor",
    "daftar",
    "deposit",
    "bonus",
    "link",
    "slot gacor",
    "gacor",
    "maxwin",
    "terpercaya",
    "tuntas",
    "ampuh",
    "terbatas",
]


def _phrase_hits(text_normalized, phrases):
    hits = []
    words = set(text_normalized.split())
    for phrase in phrases:
        phrase_norm = normalize(phrase)
        if not phrase_norm:
            continue
        if len(phrase_norm) <= 3:
            if phrase_norm in words:
                hits.append(phrase)
        elif phrase_norm in text_normalized:
            hits.append(phrase)
    return hits


def _public_interest_context(text_normalized):
    public_hits = _phrase_hits(text_normalized, PUBLIC_INTEREST_KEYWORDS)
    transaction_hits = _phrase_hits(text_normalized, TRANSACTION_KEYWORDS)
    strong_public_markers = {
        "berita",
        "berita nasional",
        "peringatan",
        "edukasi",
        "sosialisasi",
        "kampanye",
        "pencegahan",
        "anti judi",
        "jangan judi",
        "berhenti sekarang",
        "konseling",
        "investigasi",
        "kominfo",
        "pemerintah",
    }
    has_strong_public = any(h in strong_public_markers for h in public_hits)
    is_public_interest = len(public_hits) >= 2 and has_strong_public
    return is_public_interest, public_hits, transaction_hits


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
        "context_hits": [],
        "transaction_hits": [],
        "keyword_context_exempt": False,
        "keputusan": "LOLOS",
        "alasan": [],
    }

    # 1) Visual classification (CLIP zero-shot, SmolVLM, MobileNetV3, atau YOLO)
    cls_name, conf, violative, engine, detail = _visual_classify(model, image_path, engine=visual)
    result["visual_engine"] = engine
    result["yolo_class"] = cls_name
    result["yolo_conf"] = conf
    if violative:
        result["yolo_violative"] = True
        if engine == "smolvlm":
            result["alasan"].append(f"SMOLVLM menilai ILLEGAL_AD: {detail}")
        else:
            result["alasan"].append(f"{engine.upper()} deteksi kelas violative: {cls_name} (conf {conf:.2f})")

    # 2) Tesseract OCR. Untuk CLIP, OCR tetap dibaca agar poster berita/peringatan
    # tidak langsung dihukum hanya karena ada objek/kata sensitif di gambar.
    should_run_ocr = not result["yolo_violative"] or engine == "clip"
    if should_run_ocr:
        raw = ocr_text(image_path, lang=lang)
        result["ocr_text"] = raw[:500]
        public_context = False
        if raw:
            norm = normalize(raw)
            public_context, context_hits, transaction_hits = _public_interest_context(norm)
            if context_hits:
                result["context_hits"] = context_hits[:20]
            if transaction_hits:
                result["transaction_hits"] = transaction_hits[:20]

            if public_context:
                result["keyword_context_exempt"] = True
                result["alasan"].append(
                    "Konteks berita/peringatan terdeteksi: "
                    + ", ".join(context_hits[:8])
                )
                if engine == "clip" and result["yolo_violative"]:
                    result["yolo_violative"] = False
                    result["alasan"].append("Vonis CLIP ditahan karena konteks publik/edukatif")

            hits = keyword_hits(norm)
            if hits:
                result["keyword_hits"] = hits[:20]
                if public_context:
                    result["alasan"].append(
                        "OCR match keyword tetapi dikecualikan karena konteks: "
                        + ", ".join(hits[:10])
                    )
                else:
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
    effective_keyword_hit = result["keyword_hits"] and not result["keyword_context_exempt"]
    if result["yolo_violative"] or effective_keyword_hit:
        result["keputusan"] = "DIMODERASI"
    return result


def main():
    ap = argparse.ArgumentParser(description="Sistem Moderasi Gambar Otomatis")
    ap.add_argument("input", nargs="+", help="Path gambar atau folder")
    ap.add_argument("--model", default=None, help="Path model YOLO (default: best.pt dari training)")
    ap.add_argument("--visual", default=None, choices=list(VALID_VISUAL_ENGINES), help=f"Engine visual: {' | '.join(VALID_VISUAL_ENGINES)} (default: env MODERASI_VISUAL atau yolo)")
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
        conf_str = f" (conf {r['yolo_conf']:.2f})" if r["yolo_conf"] is not None else ""
        print(f"   {r['visual_engine'].upper()} : {r['yolo_class']}{conf_str}")
        if r["keyword_hits"]:
            print(f"   KEY  : {', '.join(r['keyword_hits'][:8])}")
        if r["alasan"]:
            print(f"   ALASAN: {'; '.join(r['alasan'][:3])}")

    print(f"\n=== RINGKASAN: LOLOS={summary['LOLOS']} DIMODERASI={summary['DIMODERASI']} ===")
    if args.json:
        print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
