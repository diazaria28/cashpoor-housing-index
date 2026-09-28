import argparse
import json
import re
import tempfile
import time
from pathlib import Path

import ollama
from PIL import Image

from scoring import (
    skor_air,
    skor_atap,
    skor_dinding,
    skor_lantai,
    skor_luas,
    skor_wc,
)


# =========================================================
# KONFIGURASI MODEL
# =========================================================

MODEL_DEFAULT = (
    "gemma3:4b"
)


# =========================================================
# KONFIGURASI VARIABEL
# =========================================================

KONFIGURASI = {
    "luas": {
        "pilihan": None,
        "prompt": """
Perkirakan luas bangunan utama dari foto rumah.

Aturan:
- Nilai minimum 1 m².
- Nilai maksimum 80 m².
- Abaikan halaman, jalan, kebun, dan carport terbuka.
- Jangan menghasilkan nilai lebih dari 80.
- Berikan satu angka perkiraan terbaik.
""",
    },

    "atap": {
        "pilihan": [
            "genteng_bagus",
            "genteng_beton",
            "beton",
            "multiroof",
            "genteng_biasa",
            "genteng_tanah_liat",
            "seng",
            "asbes",
            "rumbia",
            "anyaman_daun",
            "tidak_dapat_dinilai",
        ],
        "prompt": """
Identifikasi material utama atap pada foto.

Pilihan:
- genteng_bagus
- genteng_beton
- beton
- multiroof
- genteng_biasa
- genteng_tanah_liat
- seng
- asbes
- rumbia
- anyaman_daun
- tidak_dapat_dinilai

Jangan menentukan material hanya berdasarkan warna.
Perhatikan bentuk, tekstur, ketebalan, dan pola atap.
Jika material tidak terlihat jelas, pilih
tidak_dapat_dinilai.
""",
    },

    "dinding": {
        "pilihan": [
            "tembok",
            "bata",
            "batako",
            "kayu_jati_bagus",
            "setengah_tembok",
            "campuran_tembok_papan",
            "campuran_tembok_bambu",
            "kayu_biasa",
            "bambu",
            "rumbia",
            "tidak_dapat_dinilai",
        ],
        "prompt": """
Identifikasi material utama dinding pada foto.

Pilihan:
- tembok
- bata
- batako
- kayu_jati_bagus
- setengah_tembok
- campuran_tembok_papan
- campuran_tembok_bambu
- kayu_biasa
- bambu
- rumbia
- tidak_dapat_dinilai

Dinding bercat, diplester, atau diberi pelapis dekoratif
tetap termasuk tembok jika struktur utamanya tembok.
Gunakan kategori campuran jika sekitar setengah bagian
menggunakan material berbeda.
Jika tidak jelas, pilih tidak_dapat_dinilai.
""",
    },

    "lantai": {
        "pilihan": [
            "marmer",
            "keramik",
            "semen_halus",
            "semen_kasar",
            "aci",
            "bata",
            "tanah",
            "kayu",
            "tidak_dapat_dinilai",
        ],
        "prompt": """
Identifikasi material lantai utama bagian dalam rumah.

Pilihan:
- marmer
- keramik
- semen_halus
- semen_kasar
- aci
- bata
- tanah
- kayu
- tidak_dapat_dinilai

Jangan menilai halaman, jalan, teras, atau carport.
Jika lantai bagian dalam tidak terlihat jelas,
pilih tidak_dapat_dinilai.
""",
    },

    "air": {
        "pilihan": [
            "pdam",
            "air_kemasan",
            "pompa_air",
            "sumur_terlindung",
            "air_isi_ulang",
            "mata_air_terlindung",
            "sumur_tidak_terlindung",
            "air_hujan",
            "sungai",
            "danau",
            "rawa",
            "mata_air_tidak_terlindung",
            "lainnya",
            "tidak_dapat_dinilai",
        ],
        "prompt": """
Identifikasi sumber air berdasarkan objek pada foto.

Pilihan:
- pdam
- air_kemasan
- pompa_air
- sumur_terlindung
- air_isi_ulang
- mata_air_terlindung
- sumur_tidak_terlindung
- air_hujan
- sungai
- danau
- rawa
- mata_air_tidak_terlindung
- lainnya
- tidak_dapat_dinilai

Petunjuk:
- Meteran atau sambungan PDAM: pdam.
- Botol atau galon air mineral bermerek: air_kemasan.
- Mesin pompa air: pompa_air.
- Galon isi ulang tanpa merek produsen: air_isi_ulang.
- Jika sumber tidak terlihat jelas, pilih
  tidak_dapat_dinilai.
""",
    },

    "wc": {
        "pilihan": [
            "wc_duduk",
            "wc_jongkok",
            "selainnya",
            "tidak_dapat_dinilai",
        ],
        "prompt": """
Identifikasi jenis WC pada foto.

Pilihan:
- wc_duduk
- wc_jongkok
- selainnya
- tidak_dapat_dinilai

Petunjuk:
- WC dengan tempat duduk dan tangki atau dudukan:
  wc_duduk.
- WC yang digunakan dengan posisi jongkok:
  wc_jongkok.
- Fasilitas selain dua jenis tersebut: selainnya.
- Jika WC tidak terlihat jelas, pilih
  tidak_dapat_dinilai.
""",
    },
}


# =========================================================
# PERSIAPAN GAMBAR
# =========================================================

def siapkan_gambar(path_asli):
    path_asli = Path(path_asli)

    if not path_asli.exists():
        raise FileNotFoundError(
            f"Foto tidak ditemukan: {path_asli}"
        )

    with Image.open(path_asli) as image:
        image = image.convert("RGB")

        # Resolusi ringan untuk GTX 1650 4 GB.
        image.thumbnail((320, 320))

        temporary = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        )

        output = Path(temporary.name)
        temporary.close()

        image.save(
            output,
            format="JPEG",
            quality=82,
            optimize=True,
        )

    return output


# =========================================================
# JSON SCHEMA
# =========================================================

def buat_schema(task):
    if task == "luas":
        return {
            "type": "object",
            "properties": {
                "perkiraan_luas_m2": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 80,
                }
            },
            "required": [
                "perkiraan_luas_m2"
            ],
            "additionalProperties": False,
        }

    return {
        "type": "object",
        "properties": {
            "hasil": {
                "type": "string",
                "enum": KONFIGURASI[
                    task
                ]["pilihan"],
            }
        },
        "required": ["hasil"],
        "additionalProperties": False,
    }


def baca_output_model(task, output):
    """Baca JSON dan pulihkan jawaban pendek yang hanya terpotong penutupnya."""
    try:
        return json.loads(output)
    except json.JSONDecodeError as error:
        if task == "luas":
            cocok = re.search(
                r'"perkiraan_luas_m2"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                output,
            )
            if cocok:
                nilai = max(1.0, min(float(cocok.group(1)), 80.0))
                return {"perkiraan_luas_m2": nilai}
        else:
            cocok = re.search(r'"hasil"\s*:\s*"?([a-z_]+)', output)
            if cocok and cocok.group(1) in KONFIGURASI[task]["pilihan"]:
                return {"hasil": cocok.group(1)}

        raise RuntimeError(
            "Model menghasilkan JSON tidak valid: "
            f"{output}"
        ) from error


# =========================================================
# WARM-UP MODEL
# =========================================================

def warmup_model(model=MODEL_DEFAULT):
    print(
        f"Memanaskan model {model}...",
        flush=True,
    )

    waktu_mulai = time.perf_counter()

    ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Jawab hanya: OK",
            }
        ],
        options={
            "temperature": 0,
            "num_ctx": 512,
            "num_predict": 2,
        },
        keep_alive=-1,
    )

    durasi = round(
        time.perf_counter() - waktu_mulai,
        3,
    )

    print(
        f"Model siap dalam {durasi} detik.",
        flush=True,
    )


# =========================================================
# ANALISIS GAMBAR
# =========================================================

def analisis(
    task,
    image_path,
    model=MODEL_DEFAULT,
):
    if task not in KONFIGURASI:
        raise ValueError(
            f"Task tidak dikenal: {task}"
        )

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Foto tidak ditemukan: {image_path}"
        )

    gambar = siapkan_gambar(image_path)
    schema = buat_schema(task)

    prompt = f"""
Anda adalah sistem klasifikasi kondisi rumah.

Variabel:
{task}

{KONFIGURASI[task]["prompt"]}

Aturan wajib:
- Gunakan hanya objek yang terlihat pada gambar.
- Jangan memberikan penjelasan.
- Jangan menulis teks di luar JSON.
- Jawaban harus sangat singkat.
- Jika ragu, gunakan tidak_dapat_dinilai,
  kecuali untuk perkiraan luas.
"""

    print(
        f"Mengirim task '{task}' "
        f"ke {model}...",
        flush=True,
    )

    waktu_mulai = time.perf_counter()

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(gambar)],
                }
            ],
            format=schema,
            options={
                "temperature": 0,
                "num_ctx": 512,
                "num_predict": 32,
                "top_k": 1,
                "seed": 42,
            },
            keep_alive=-1,
        )

        durasi = round(
            time.perf_counter()
            - waktu_mulai,
            3,
        )

        print(
            "Respons model diterima.",
            flush=True,
        )

        output = (
            response.message.content.strip()
        )

        data = baca_output_model(task, output)

        return {
            "task": task,
            "model": model,
            "prediksi": data,
            "durasi_detik": durasi,
        }

    finally:
        try:
            gambar.unlink(
                missing_ok=True
            )
        except OSError:
            pass


# =========================================================
# PERHITUNGAN SKOR
# =========================================================

def tambahkan_skor(
    hasil,
    jumlah_anggota=None,
):
    task = hasil["task"]
    prediksi = hasil["prediksi"]

    if task == "luas":
        if jumlah_anggota is None:
            raise ValueError(
                "Jumlah anggota keluarga "
                "wajib untuk skor luas."
            )

        perkiraan_luas = prediksi[
            "perkiraan_luas_m2"
        ]

        hasil_luas = skor_luas(
            perkiraan_luas,
            jumlah_anggota,
        )

        hasil["prediksi"] = hasil_luas
        hasil["skor"] = hasil_luas["skor"]

        return hasil

    hasil_ai = prediksi["hasil"]

    fungsi_skor = {
        "atap": skor_atap,
        "dinding": skor_dinding,
        "lantai": skor_lantai,
        "air": skor_air,
        "wc": skor_wc,
    }

    hasil["skor"] = fungsi_skor[
        task
    ](hasil_ai)

    return hasil


# =========================================================
# COMMAND LINE
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analisis satu variabel rumah"
        )
    )

    parser.add_argument(
        "--task",
        required=True,
        choices=list(
            KONFIGURASI.keys()
        ),
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Lokasi foto",
    )

    parser.add_argument(
        "--members",
        type=int,
        default=None,
        help=(
            "Jumlah anggota keluarga. "
            "Wajib untuk task luas."
        ),
    )

    parser.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        help="Nama model Ollama",
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help=(
            "Lewati warm-up jika model "
            "sudah aktif."
        ),
    )

    args = parser.parse_args()

    if (
        args.task == "luas"
        and args.members is None
    ):
        parser.error(
            "--members wajib untuk task luas"
        )

    if (
        args.members is not None
        and args.members <= 0
    ):
        parser.error(
            "--members minimal 1"
        )

    if not args.skip_warmup:
        warmup_model(args.model)

    hasil = analisis(
        task=args.task,
        image_path=args.image,
        model=args.model,
    )

    hasil = tambahkan_skor(
        hasil,
        jumlah_anggota=args.members,
    )

    print(
        json.dumps(
            hasil,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
