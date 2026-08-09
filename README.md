# Moderasi Gambar Otomatis (Satudata)

Sistem moderasi gambar otomatis yang mendeteksi konten ilegal/berbahaya menggunakan **engine visual** (YOLO11-cls, MobileNetV3, atau CLIP) + **Tesseract OCR** (ekstraksi teks) + **keyword filtering** (blacklist) + **context gate** untuk membedakan iklan/promosi dari berita, edukasi, atau peringatan publik. Keputusan akhir: **LOLOS** atau **DIMODERASI**.

## Kelas Model

| Kelas | Aksi |
|---|---|
| `obat_aborsi` | DIMODERASI |
| `normal` | LOLOS |
| `dokumen` | LOLOS |

## Arsitektur

```
Input Gambar
  → pilih engine visual: YOLO / MobileNetV3 / CLIP
  → jika YOLO/MobileNetV3 violative conf tinggi: DIMODERASI (OCR dilewati)
  → jika CLIP violative: lanjut OCR + Context Gate
  → jika visual tidak violative: lanjut OCR + Keyword Filter

OCR + Keyword + Context Gate
  → keyword cocok + konteks promosi/transaksi: DIMODERASI
  → keyword cocok + konteks berita/edukasi/peringatan: LOLOS
  → tidak ada keyword cocok: LOLOS
```

Keputusan **DIMODERASI** jika YOLO/MobileNetV3 mendeteksi kelas violative dengan konfidensi tinggi, atau OCR menemukan keyword blacklist dalam konteks promosi/transaksi. Untuk **CLIP**, hasil violative tidak lagi langsung final: OCR tetap dijalankan untuk mengecek apakah gambar adalah berita, edukasi, peringatan, kampanye publik, atau anti-judi. Keyword seperti `aborsi`, `boraks`, atau `judi online` bisa dikecualikan jika konteksnya jelas publik/edukatif, bukan jualan/promosi.

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
- `GET  /api/engines` — daftar engine visual
- `GET  /health` — status server & kelas model

Respons moderasi menyertakan waktu proses:

- `ringkasan.elapsed_ms` — total waktu request dalam milidetik.
- `ringkasan.avg_elapsed_ms` — rata-rata waktu proses per gambar.
- `results[].elapsed_ms` — waktu proses tiap gambar.

## CLI Offline

```sh
uv run python moderasi.py <gambar atau folder> [--json]
```

## Keterbatasan

- Dataset kecil & noisy (hasil scrape) — akurasi bisa ditingkatkan dengan data tambahan.
- OCR (~550ms/gambar) jadi bottleneck untuk gambar yang perlu analisis teks. Gambar besar otomatis di-resize ke max 1600px sebelum OCR. Untuk CLIP violative, OCR tetap dijalankan agar poster berita/edukasi tidak langsung salah-moderasi.
- Membutuhkan **Tesseract-OCR** terinstall (default `C:\Program Files\Tesseract-OCR\tesseract.exe`).
