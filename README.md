# Moderasi Gambar Otomatis (Satudata)

Sistem moderasi gambar otomatis yang mendeteksi konten ilegal/berbahaya menggunakan **engine visual** (YOLO11-cls, MobileNetV3, CLIP, atau SmolVLM2-500M) + **Tesseract OCR** (ekstraksi teks) + **keyword filtering** (blacklist) + **context gate** untuk membedakan iklan/promosi dari berita, edukasi, atau peringatan publik. Keputusan akhir: **LOLOS** atau **DIMODERASI**.

## Kelas Model

| Kelas | Aksi |
|---|---|
| `obat_aborsi` | DIMODERASI |
| `normal` | LOLOS |
| `dokumen` | LOLOS |

## Arsitektur

```
Input Gambar
  → pilih engine visual: YOLO / MobileNetV3 / CLIP / SmolVLM2
  → jika YOLO/MobileNetV3 violative conf tinggi: DIMODERASI (OCR dilewati)
  → jika CLIP violative: tetap lanjut OCR + Context Gate
  → jika SmolVLM menilai ILLEGAL_AD: DIMODERASI (OCR dilewati)
  → jika SmolVLM menilai PUBLIC_INFO/NORMAL: lanjut OCR + Keyword Filter
  → jika visual aman/ragu: lanjut OCR + Keyword Filter

OCR + Keyword + Context Gate
  → keyword cocok + konteks promosi/transaksi: DIMODERASI
  → keyword cocok + konteks berita/edukasi/peringatan: LOLOS
  → tidak ada keyword cocok: LOLOS
```

Keputusan **DIMODERASI** jika YOLO/MobileNetV3 mendeteksi kelas violative dengan konfidensi tinggi, SmolVLM menilai gambar sebagai `ILLEGAL_AD`, atau OCR menemukan keyword blacklist dalam konteks promosi/transaksi. Untuk **CLIP**, hasil violative tidak langsung final: OCR tetap dijalankan untuk mengecek apakah gambar adalah berita, edukasi, peringatan, kampanye publik, atau anti-judi. Untuk **SmolVLM** yang menjawab `PUBLIC_INFO` atau `NORMAL`, OCR tetap dipakai sebagai safety net agar teks promosi yang lolos dari VLM masih bisa tertangkap keyword/context gate.

Jika engine pilihan gagal saat runtime, server mencatat warning dan fallback ke YOLO agar request tetap mendapat hasil. Field `visual_engine` pada response menunjukkan engine yang benar-benar dipakai.

## Pilihan Engine Visual

| Engine | Karakter | Cocok untuk |
|---|---|---|
| `yolo` | Paling ringan dan cepat; memakai model lokal `models/best.pt`. | Default CPU, batch besar, respons cepat. |
| `mobilenetv3` | Ringan, model lokal `models/mobilenetv3_best.pt`, akurasi validasi lebih baik pada dataset ini. | Default produksi jika model tersedia. |
| `clip` | Zero-shot; lebih fleksibel untuk gambar di luar dataset, tetapi lebih berat dari YOLO/MobileNetV3. | Uji kasus out-of-distribution dan poster unik. |
| `smolvlm` | Vision-language model; bisa membaca konteks gambar dan teks, tetapi download/load/inference paling lambat. | Review manual gambar sulit, bukan batch besar. |

### Catatan SmolVLM

SmolVLM lambat karena ada tiga tahap berbeda:

1. **Download model** — terjadi saat pertama kali engine `smolvlm` dipakai jika cache Hugging Face belum ada. Ukuran model dan file pendukungnya besar, dan tanpa `HF_TOKEN` download bisa lebih lambat/terbatas.
2. **Load model** — terjadi setiap server baru dinyalakan dan `smolvlm` pertama kali dipakai. Bobot dimuat ke RAM/VRAM.
3. **Inference** — proses membaca gambar dan menghasilkan jawaban. Di CPU, ini bisa belasan sampai puluhan detik per gambar.

Gunakan `HF_HOME` untuk cache model, misalnya:

```sh
HF_HOME=/opt/moderasi/hf_cache MODERASI_VISUAL=smolvlm uv run python server.py
```

Untuk penggunaan harian, rekomendasi default adalah `mobilenetv3` atau `yolo`, lalu OCR + keyword/context gate menangani iklan berbasis teks. Pakai `smolvlm` hanya saat butuh analisis gambar yang sulit dan jumlahnya sedikit.

## Instalasi (uv)

```sh
uv sync          # buat .venv + install deps (dari pyproject.toml/uv.lock)
```

Alternatif pip:

```sh
pip install -r requirements.txt
```

Dependency opsional:

- `clip` dan `smolvlm` butuh `transformers` serta PyTorch.
- `mobilenetv3` butuh `torchvision` dan file `models/mobilenetv3_best.pt`.
- Di deployment VPS, `deploy/deploy.sh` memasang PyTorch CPU dan hanya memasang `transformers` saat `VISUAL_ENGINE=clip` atau `VISUAL_ENGINE=smolvlm`.

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

Pilih engine default saat start:

```sh
MODERASI_VISUAL=mobilenetv3 uv run python server.py
MODERASI_VISUAL=clip uv run python server.py
MODERASI_VISUAL=smolvlm uv run python server.py
```

Atau pilih per request:

```sh
curl -F "file=@foto.jpg" "http://127.0.0.1:8787/api/moderasi/satu?visual=clip"
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
- SmolVLM tidak disarankan untuk batch besar di CPU. Request pertama bisa lama karena download/cache model, dan request pertama setelah server restart bisa lama karena load model.
- CLIP/SmolVLM memakai model Hugging Face. Tanpa cache stabil (`HF_HOME`) atau koneksi baik, request pertama bisa terlihat lambat.
- Membutuhkan **Tesseract-OCR** terinstall (default `C:\Program Files\Tesseract-OCR\tesseract.exe`).
