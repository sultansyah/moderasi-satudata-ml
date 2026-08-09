#!/usr/bin/env bash
#
# Deploy Script - Sistem Moderasi Gambar Otomatis
# Target : Ubuntu / Debian (VPS), native install + systemd
# Cara   : dari folder proyek ini:
#          sudo bash deploy/deploy.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "deploy" ]]; then
  SRC_DIR="$(dirname "$SCRIPT_DIR")"
else
  SRC_DIR="$SCRIPT_DIR"
fi
APP_DIR="/opt/moderasi"
SERVICE="moderasi.service"
PORT="${PORT:-8787}"
# Engine visual default (yolo | clip | mobilenetv3 | smolvlm). Set sekali di sini, disuntikkan
# ke systemd service:  sudo VISUAL_ENGINE=mobilenetv3 bash deploy/deploy.sh
VISUAL_ENGINE="${VISUAL_ENGINE:-yolo}"
case "$VISUAL_ENGINE" in
  yolo|clip|mobilenetv3|smolvlm) ;;
  *) echo "[ERROR] VISUAL_ENGINE tidak valid: $VISUAL_ENGINE (pilihan: yolo, clip, mobilenetv3, smolvlm)"; exit 1 ;;
esac

if [[ "$EUID" -ne 0 ]]; then
  echo "[ERROR] Jalankan sebagai root: sudo bash deploy.sh"
  exit 1
fi

echo "==> [1/7] Install paket sistem (Python + Tesseract + bahasa Indonesia)"
apt-get update -y
apt-get install -y python3 python3-venv python3-pip \
  tesseract-ocr tesseract-ocr-ind tesseract-ocr-vie \
  libgl1 libglib2.0-0 curl

echo "==> [2/7] Salin file aplikasi ke $APP_DIR"
mkdir -p "$APP_DIR"
for f in server.py moderasi.py keywords.py ui.html requirements.txt; do
  [[ -f "$SRC_DIR/$f" ]] || { echo "[ERROR] $f tidak ditemukan di $SRC_DIR"; exit 1; }
  cp -f "$SRC_DIR/$f" "$APP_DIR/"
done
# model utama (models/best.pt) - di-commit ke repo
if [[ -f "$SRC_DIR/models/best.pt" ]]; then
  mkdir -p "$APP_DIR/models"
  cp -f "$SRC_DIR/models/best.pt" "$APP_DIR/models/"
else
  echo "[WARN] models/best.pt tidak ada - pastikan model tersedia nanti"
fi
# model alternatif MobileNetV3 (untuk MODERASI_VISUAL=mobilenetv3)
if [[ -f "$SRC_DIR/models/mobilenetv3_best.pt" ]]; then
  mkdir -p "$APP_DIR/models"
  cp -f "$SRC_DIR/models/mobilenetv3_best.pt" "$APP_DIR/models/"
fi
# fallback: model dari hasil training lokal (runs/) kalau ada
if [[ -d "$SRC_DIR/runs" ]]; then
  cp -rf "$SRC_DIR/runs" "$APP_DIR/"
fi

echo "==> [3/7] Buat virtual environment Python"
cd "$APP_DIR"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -q

echo "==> [4/7] Install PyTorch versi CPU (menghindari unduhan CUDA ~2GB)"
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "==> [5/7] Install dependensi aplikasi"
pip install -q -r requirements.txt
# Engine berbasis transformers (clip / smolvlm) butuh transformers
if [[ "$VISUAL_ENGINE" == "clip" || "$VISUAL_ENGINE" == "smolvlm" ]]; then
  echo "    (engine=$VISUAL_ENGINE) install transformers..."
  pip install -q transformers
fi
if [[ "$VISUAL_ENGINE" == "clip" ]]; then
  echo "    (engine=clip) unduh bobot CLIP (~350MB)..."
  python - <<'PY'
import os
os.environ.setdefault("HF_HOME", "/opt/moderasi/hf_cache")
from transformers import CLIPModel, CLIPProcessor
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("    CLIP model siap.")
PY
fi
if [[ "$VISUAL_ENGINE" == "smolvlm" ]]; then
  echo "    (engine=smolvlm) install transformers + num2words, unduh bobot (~1GB)..."
  pip install -q transformers num2words
  python - <<'PY'
import os
os.environ.setdefault("HF_HOME", "/opt/moderasi/hf_cache")
from transformers import AutoModelForImageTextToText, AutoProcessor
AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")
AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")
print("    SmolVLM model siap.")
PY
fi

echo "==> [6/7] Pasang systemd service ($SERVICE)"
if [[ "$VISUAL_ENGINE" == "yolo" ]]; then
  sed "s/--port 8787/--port ${PORT}/" "$SRC_DIR/deploy/$SERVICE" > "/etc/systemd/system/$SERVICE"
else
  sed -e "s/--port 8787/--port ${PORT}/" \
      -e "/Environment=PYTHONUNBUFFERED=1/a Environment=MODERASI_VISUAL=${VISUAL_ENGINE}" \
      "$SRC_DIR/deploy/$SERVICE" > "/etc/systemd/system/$SERVICE"
  echo "    MODERASI_VISUAL=${VISUAL_ENGINE} ditambahkan ke service."
fi
systemctl daemon-reload
systemctl enable "$SERVICE"
systemctl restart "$SERVICE"

echo "==> [7/7] Verifikasi"
sleep 3
if systemctl is-active --quiet "$SERVICE"; then
  echo "    Service AKTIF ✓"
else
  echo "    Service GAGAL - cek log: journalctl -u $SERVICE -e"
fi
curl -sf "http://127.0.0.1:${PORT}/health" && echo "    /health OK ✓" || echo "    /health belum merespons"

echo ""
echo "=========================================================="
echo "  SELESAI! Server moderasi berjalan."
echo "  URL        : http://IP_VPS:${PORT}/"
echo "  API        : POST http://IP_VPS:${PORT}/api/moderasi/bulk"
echo "  Status     : systemctl status $SERVICE"
echo "  Log        : journalctl -u $SERVICE -f"
echo "  Restart    : systemctl restart $SERVICE"
echo "  Engine     : $VISUAL_ENGINE (ganti default via UI atau POST /api/engines/default?engine=...)"
echo "  Catatan    : buka port ${PORT} di firewall (ufw allow ${PORT}/tcp)"
echo "=========================================================="
