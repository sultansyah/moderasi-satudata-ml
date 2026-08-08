#!/usr/bin/env bash
#
# Deploy Script - Sistem Moderasi Gambar Otomatis
# Target : Ubuntu / Debian (VPS), native install + systemd
# Cara   : jalankan dari folder proyek ini:
#          sudo bash deploy.sh
#
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="/opt/moderasi"
SERVICE="moderasi.service"
PORT="${PORT:-8787}"

if [[ "$EUID" -ne 0 ]]; then
  echo "[ERROR] Jalankan sebagai root: sudo bash deploy.sh"
  exit 1
fi

echo "==> [1/7] Install paket sistem (Python + Tesseract + bahasa Indonesia)"
apt-get update -y
apt-get install -y python3 python3-venv python3-pip \
  tesseract-ocr tesseract-ocr-ind \
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

echo "==> [6/7] Pasang systemd service ($SERVICE)"
sed "s/--port 8787/--port ${PORT}/" "$SRC_DIR/deploy/$SERVICE" > "/etc/systemd/system/$SERVICE"
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
echo "  Catatan    : buka port ${PORT} di firewall (ufw allow ${PORT}/tcp)"
echo "=========================================================="
