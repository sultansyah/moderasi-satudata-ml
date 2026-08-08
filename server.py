import os
import sys
import tempfile
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from PIL import Image

from moderasi import load_model, moderasi_satu_gambar, KELAS_VIOLATIVE
from keywords import KEYWORDS_ALL, KEYWORDS_ABORSI, KEYWORDS_BORAKS, KEYWORDS_UMUM

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
MAX_WORKERS = 4

app = FastAPI(
    title="Sistem Moderasi Gambar Otomatis",
    description="Moderasi gambar via YOLO11 (klasifikasi). Jika violative → langsung DIMODERASI; "
                "jika normal → Tesseract OCR + keyword filtering. Keputusan: LOLOS / DIMODERASI.",
    version="1.0.0",
)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def _simpan_temp(name, data):
    ext = os.path.splitext(name)[1].lower() or ".jpg"
    if ext not in ALLOWED_EXT:
        ext = ".jpg"
    path = os.path.join(tempfile.gettempdir(), f"moderasi_{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(data)
    return path


def _moderate_bytes(name, data):
    tmp = _simpan_temp(name, data)
    try:
        try:
            with Image.open(tmp) as im:
                im.load()
        except Exception as e:
            return {
                "file": name,
                "filename": name,
                "keputusan": "ERROR",
                "alasan": [f"Bukan gambar valid: {e}"],
                "yolo_class": None,
                "yolo_conf": None,
                "keyword_hits": [],
                "ocr_text": "",
            }
        result = moderasi_satu_gambar(get_model(), tmp)
        result["file"] = name
        result["filename"] = name
        return result
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _ringkasan(results):
    c = Counter(r["keputusan"] for r in results)
    return {
        "total": len(results),
        "dimoderasi": c.get("DIMODERASI", 0),
        "lolos": c.get("LOLOS", 0),
        "error": c.get("ERROR", 0),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
    if os.path.isfile(ui_path):
        return FileResponse(ui_path, media_type="text/html")
    return HTMLResponse("<h3>ui.html tidak ditemukan di samping server.py</h3>")


@app.post("/api/moderasi/satu")
async def moderasi_satu(file: UploadFile = File(...)):
    """Moderasi SATU gambar. Upload via multipart form (field: file)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nama file kosong")
    data = await file.read()
    result = _moderate_bytes(file.filename, data)
    return {"ringkasan": _ringkasan([result]), "results": [result]}


@app.post("/api/moderasi/bulk")
async def moderasi_bulk(files: list[UploadFile] = File(...)):
    """Moderasi BANYAK gambar sekaligus. Upload via multipart form (field: files)."""
    if not files:
        raise HTTPException(status_code=400, detail="Tidak ada file diunggah")
    if len(files) > 1000:
        raise HTTPException(status_code=400, detail="Maksimal 1000 file per request")

    items = []
    for f in files:
        data = await f.read()
        items.append((f.filename or f"unnamed_{len(items)}.jpg", data))

    if len(items) == 1:
        results = [_moderate_bytes(*items[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(items))) as ex:
            results = list(ex.map(lambda t: _moderate_bytes(*t), items))

    return {"ringkasan": _ringkasan(results), "results": results}


@app.get("/api/keywords")
def get_keywords():
    return {
        "total": len(KEYWORDS_ALL),
        "aborsi": sorted(KEYWORDS_ABORSI),
        "boraks": sorted(KEYWORDS_BORAKS),
        "umum": sorted(KEYWORDS_UMUM),
    }


@app.get("/health")
def health():
    m = get_model()
    return {"status": "ok", "model": os.path.basename(m.model.pt_path or "best.pt"), "kelas": list(m.names.values())}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Web Server Moderasi Gambar Otomatis")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--model", default=None, help="Path model YOLO (override default)")
    args = ap.parse_args()

    if args.model:
        from moderasi import DEFAULT_MODEL
        os.environ["MODERASI_MODEL"] = os.path.abspath(args.model)

    print(f"=== Server Moderasi Gambar ===\n  http://{args.host}:{args.port}\n"
          f"  GET  /               -> halaman upload\n"
          f"  POST /api/moderasi/satu  (field: file)\n"
          f"  POST /api/moderasi/bulk  (field: files)\n"
          f"  GET  /health\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
