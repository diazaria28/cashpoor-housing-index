# Cashpoor Housing Index (CHI) — Sistem Penilaian Kondisi Rumah

Prototipe sistem berbasis AI untuk membantu Account Officer menilai kondisi fisik rumah melalui foto. Sistem menganalisis komponen atap, dinding, lantai, sumber air, dan WC, kemudian memberikan kategori serta rekomendasi skor.

> **Status:** prototipe penelitian. Hasil AI wajib diperiksa dan dapat dikoreksi oleh petugas. Sistem tidak boleh menjadi satu-satunya dasar pengambilan keputusan pembiayaan.

## Fitur

- Validasi kesesuaian foto dengan kolom penilaian.
- Klasifikasi kondisi atap.
- Klasifikasi material dinding.
- Klasifikasi kondisi lantai.
- Klasifikasi sumber air.
- Klasifikasi jenis WC.
- Analisis tambahan untuk material dinding campuran.
- Perhitungan skor sementara.
- Koreksi hasil oleh petugas.
- Antarmuka web lokal.
- Pemrosesan model secara lokal.

## Arsitektur model

### OpenCLIP

Model utama:

```text
Arsitektur : ViT-B-32
Pretrained : laion2b_s34b_b79k
Framework  : open-clip-torch
```

OpenCLIP digunakan untuk:

1. Memastikan foto sesuai dengan komponen yang dipilih.
2. Membandingkan fitur visual foto dengan kumpulan prompt setiap kategori.
3. Menghasilkan probabilitas relatif dan kategori kondisi rumah.

Klasifikasi OpenCLIP menggunakan pendekatan zero-shot sehingga tidak memerlukan pelatihan ulang pada dataset proyek.

### Grounding DINO

Grounding DINO digunakan sebagai pemeriksa tambahan khusus dinding, terutama ketika OpenCLIP memprediksi:

- `setengah_tembok`
- `kayu_bambu`

Model ini membantu menemukan keberadaan material tembok dan kayu/bambu menggunakan text-guided object detection. Grounding DINO tidak menggantikan OpenCLIP untuk seluruh kategori.

### Gemma melalui Ollama

Model `gemma3:4b` digunakan pada fitur eksperimen estimasi luas bangunan dari foto.

Estimasi luas masih bersifat indikatif dan wajib diverifikasi petugas. Hasil evaluasi menunjukkan bahwa estimasi satu foto belum cukup akurat untuk digunakan sebagai ukuran pasti atau komponen skor otomatis.

## Kategori dan skor

| Komponen | Kategori | Skor |
|---|---|---:|
| Atap | Genteng bagus/beton/multiroof | 3 |
| Atap | Genteng tanah liat/seng/asbes | 1 |
| Atap | Rumbia/anyaman daun | 0 |
| Dinding | Tembok penuh | 3 |
| Dinding | Setengah tembok/campuran | 1 |
| Dinding | Kayu/bambu/rumbia | 0 |
| Lantai | Marmer/keramik | 3 |
| Lantai | Semen/aci/bata | 1 |
| Lantai | Tanah/kayu | 0 |
| Air | PDAM/air mineral kemasan | 3 |
| Air | Pompa/sumur terlindung/isi ulang | 1 |
| Air | Sumber air tidak layak | 0 |
| WC | WC duduk | 3 |
| WC | WC jongkok | 1 |
| WC | Selain WC duduk/jongkok | 0 |

## Persyaratan

- Windows 10 atau Windows 11
- Python 3.11
- Git
- Ollama
- Model Ollama `gemma3:4b`
- GPU NVIDIA dengan CUDA direkomendasikan, tetapi CPU tetap dapat digunakan dengan proses yang lebih lambat

## Instalasi

Clone repository:

```powershell
git clone https://github.com/USERNAME/cashpoor-housing-index.git
cd cashpoor-housing-index
```

Buat virtual environment menggunakan Python 3.11:

```powershell
py -3.11 -m venv .venv_yolo
```

Instal dependensi tanpa harus mengaktifkan virtual environment:

```powershell
.\.venv_yolo\Scripts\python.exe -m pip install --upgrade pip
.\.venv_yolo\Scripts\python.exe -m pip install -r requirements.txt
```

Pastikan Ollama sudah berjalan dan unduh model Gemma:

```powershell
ollama pull gemma3:4b
ollama list
```

Bobot atau konfigurasi model lain dapat diunduh otomatis oleh library saat pertama kali digunakan. Koneksi internet mungkin dibutuhkan pada eksekusi pertama.

## Menjalankan aplikasi

Cara termudah di Windows:

```powershell
.\jalankan_web.bat
```

Atau jalankan secara manual:

```powershell
.\.venv_yolo\Scripts\python.exe -m uvicorn experiments.api_web_hybrid:app --host 127.0.0.1 --port 7863
```

Buka alamat berikut:

```text
http://127.0.0.1:7863
```

Hentikan server dengan menekan `Ctrl+C`.

## Struktur utama

```text
cashpoor-housing-index/
├── analyzer.py
├── clip_analyzer.py
├── scoring.py
├── requirements.txt
├── jalankan_web.bat
├── baseline_final/
│   ├── app_clip.py
│   ├── clip_analyzer.py
│   ├── evaluate_dataset_folder.py
│   └── buat_confusion_matrix.py
├── experiments/
│   ├── api_web_hybrid.py
│   ├── dinding_grounding_dino/
│   │   └── dinding_grounding_dino.py
│   └── dinding_hybrid/
│       └── dinding_hybrid.py
└── web_ui_hybrid/
    ├── index.html
    ├── app.js
    ├── styles.css
    └── PNM_logo.svg
```

## Hasil evaluasi klasifikasi utama

Evaluasi dilakukan pada:

- 117 foto sesuai kolom.
- 100 foto salah kolom.
- Total 217 foto.

Ringkasan hasil:

| Metrik | Nilai |
|---|---:|
| Akurasi label | 88,89% |
| Akurasi skor | 88,89% |
| MAE skor | 0,115 |
| False Acceptance Rate | 0,00% |
| False Rejection Rate | 3,42% |
| Median durasi | 0,0446 detik |
| P95 durasi | 0,0823 detik |

Hasil per komponen:

| Komponen | Akurasi |
|---|---:|
| Atap | 100,00% |
| Dinding | 62,96% |
| Lantai | 100,00% |
| Air | 92,00% |
| WC | 95,45% |

Nilai tersebut hanya berlaku untuk dataset evaluasi yang digunakan dan tidak mewakili seluruh variasi rumah di Indonesia.

## Evaluasi model dinding

Perbandingan dilakukan pada 32 foto dinding.

| Metode | Akurasi label | Akurasi operasional | Macro F1 | Median waktu |
|---|---:|---:|---:|---:|
| OpenCLIP | 68,75% | 62,50% | 0,6548 | 0,378 detik |
| Grounding DINO | 56,25% | — | 0,6222 | 8,833 detik |
| Hybrid | 84,38% | 78,12% | 0,8481 | 9,2968 detik |

Pendekatan hybrid meningkatkan hasil klasifikasi dinding, tetapi membutuhkan waktu inferensi lebih lama.

## Evaluasi estimasi luas

Evaluasi eksperimen dilakukan pada 535 foto Houses Dataset.

| Model | Coverage | MAE | MAPE |
|---|---:|---:|---:|
| Gemma3:4B | 100,00% | 79,91 m² | 34,25% |
| YOLOE pintu–fasad | 48,04% | 120,15 m² | 52,98% |

Pada 257 foto yang menghasilkan prediksi dari kedua metode, Gemma lebih dekat pada 189 foto dan YOLOE lebih dekat pada 68 foto.

Namun, hasil tersebut belum dapat menjadi validasi final karena:

- 98,13% label dataset berada di atas batas keluaran Gemma sebesar 80 m².
- Label dataset merepresentasikan luas listing atau luas total.
- Estimasi YOLOE dari satu foto lebih dekat dengan perkiraan luas tapak.
- Dataset berasal dari rumah di Amerika dan berbeda dari karakteristik rumah Indonesia.
- Kedalaman bangunan tidak dapat diketahui secara pasti melalui satu foto tampak depan.

Oleh karena itu, estimasi luas tidak digunakan sebagai ukuran pasti atau dasar skor otomatis.

## Dataset dan bobot model

Dataset, foto pengujian, virtual environment, hasil inferensi, serta bobot model tidak disertakan dalam repository karena alasan ukuran, privasi, dan lisensi.

File yang dikecualikan antara lain:

```text
dataset/
foto/
weights/
.venv/
.venv_yolo/
hasil/
hasil_*/
*.pt
*.pth
*.ts
*.onnx
*.safetensors
```

Pengguna harus menyediakan foto pengujian sendiri dan memastikan bahwa penggunaan data sudah memperoleh izin yang sesuai.

## Keterbatasan

- Dataset evaluasi masih relatif kecil.
- Sebagian foto dapat memiliki pencahayaan, sudut, atau kualitas yang berbeda dari data pengujian.
- Model zero-shot sensitif terhadap pemilihan prompt.
- Kategori dinding campuran masih menjadi kasus paling sulit.
- Model tambahan dinding meningkatkan akurasi, tetapi memperlambat proses.
- Estimasi luas dari satu foto belum cukup akurat.
- Hasil model dapat salah meskipun nilai keyakinannya tinggi.
- Sistem belum divalidasi sebagai alat keputusan pembiayaan.

## Penggunaan yang disarankan

Sistem digunakan sebagai alat bantu survei dengan prinsip **human-in-the-loop**:

1. Petugas mengunggah foto komponen rumah.
2. Sistem melakukan validasi dan klasifikasi.
3. Sistem memberikan rekomendasi kategori dan skor.
4. Petugas memeriksa foto dan hasil AI.
5. Petugas dapat mengoreksi hasil sebelum menyimpan penilaian.

## Etika dan privasi

Jangan mengunggah foto nasabah atau informasi pribadi ke repository publik. Pastikan foto yang diproses telah mendapatkan persetujuan dan digunakan sesuai kebijakan perlindungan data yang berlaku.

## Lisensi dan penggunaan logo

Kode sumber dan aset memiliki ketentuan penggunaan masing-masing. Logo PNM merupakan identitas perusahaan dan tidak otomatis menjadi bagian dari lisensi kode sumber. Pastikan izin penggunaan logo telah diperoleh sebelum repository dipublikasikan.

## Penafian

Proyek ini merupakan prototipe penelitian dan demonstrasi teknis. Hasil AI bersifat rekomendasi, bukan keputusan final. Pemeriksaan lapangan dan keputusan petugas tetap menjadi sumber penilaian utama.