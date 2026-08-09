# Moderasi Gambar Otomatis (Satudata)

Sistem moderasi gambar otomatis yang mendeteksi konten ilegal/berbahaya menggunakan **YOLO11-cls** (klasifikasi visual) + **Tesseract OCR** (ekstraksi teks) + **keyword filtering** (blacklist). Keputusan akhir: **LOLOS** atau **DIMODERASI**.

## Kelas Model

| Kelas | Aksi |
|---|---|
| `obat_aborsi` | DIMODERASI |
| `normal` | LOLOS |

## Arsitektur

```
Input Gambar → YOLO11-cls ──violative──→ DIMODERASI (OCR dilewati)
                      └─ normal → Tesseract OCR → Keyword Filter → DIMODERASI
                                                        └─ hasil negatif → LOLOS
```

Keputusan **DIMODERASI** jika YOLO mendeteksi kelas violative (langsung, tanpa OCR) **ATAU** OCR/cocok dengan keyword blacklist (aborsi, cytotec, misoprostol, boraks, dsb.). OCR hanya dijalankan saat YOLO = `normal`.

## Instalasi (uv)

```sh
uv sync          # buat .venv + install deps (dari pyproject.toml/uv.lock)
```

Alternatif pip:

```sh
pip install -r requirements.txt
```

## Menyiapkan Dataset

Struktur folder sumber di luar repo, mis. `dataset_obat_aborsi_google/`, `dataset_normal/`. Bangun dataset train/val:

```sh
uv run python prepare_dataset.py
```

> Edit `SOURCES` di `prepare_dataset.py` sesuai lokasi folder sumber. Dataset & model tidak di-commit ke git (lihat `.gitignore`).

## Training

```sh
uv run python train.py
```

Model terbaik tersimpan di `runs/yolo11n-cls-mod-v4/weights/best.pt`. Salin ke `models/best.pt` untuk dipakai server.

## Menjalankan Server Web

```sh
uv run python server.py --host 127.0.0.1 --port 8787
```

- `GET  /` — halaman upload (`ui.html`)
- `POST /api/moderasi/satu` — moderasi 1 gambar (field: `file`)
- `POST /api/moderasi/bulk` — moderasi banyak gambar (field: `files`, max 1000)
- `GET  /api/keywords` — daftar keyword blacklist (aborsi/boraks/judi/umum)
- `GET  /health` — status server & kelas model

## CLI Offline

```sh
uv run python moderasi.py <gambar atau folder> [--json]
```

## Keterbatasan

- Dataset kecil & noisy (hasil scrape) — akurasi bisa ditingkatkan dengan data tambahan.
- OCR (~550ms/gambar) jadi bottleneck untuk gambar `normal`; YOLO hanya ~37ms. Gambar besar otomatis di-resize ke max 1600px sebelum OCR, dan OCR otomatis dilewati saat YOLO = violative.
- Membutuhkan **Tesseract-OCR** terinstall (default `C:\Program Files\Tesseract-OCR\tesseract.exe`).
