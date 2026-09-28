import json
from datetime import datetime
from pathlib import Path

import gradio as gr

from analyzer import MODEL_DEFAULT, analisis, tambahkan_skor
from clip_analyzer import KATEGORI, MODEL_NAME, PRETRAINED, klasifikasi


BASE_FOLDER = Path(__file__).resolve().parent
OUTPUT_FOLDER = BASE_FOLDER / "hasil_clip"
OUTPUT_FOLDER.mkdir(exist_ok=True)

TASKS = ["atap", "dinding", "lantai", "air", "wc"]


def pilihan_dropdown(task):
    return [
        (f'{data["nama"]} — skor {data["skor"]}', key)
        for key, data in KATEGORI[task].items()
    ]


def skor_pilihan(task, value):
    if not value or value not in KATEGORI[task]:
        return None

    return KATEGORI[task][value]["skor"]


def analisis_clip_ui(task, image_path):
    if not image_path:
        raise gr.Error("Ambil atau unggah foto terlebih dahulu.")

    try:
        hasil = klasifikasi(task, image_path)
    except Exception as error:
        raise gr.Error(f"Analisis foto gagal: {error}") from error

    validasi = hasil.get("validasi_foto", {})
    status_hasil = hasil.get("status")

    if status_hasil == "foto_tidak_sesuai":
        terdeteksi = validasi.get(
            "komponen_terdeteksi",
            "objek lain",
        )

        status = (
            f"❌ **Foto ditolak.** Kolom ini membutuhkan foto **{task}**, "
            f"tetapi gambar lebih menyerupai **{terdeteksi}**.  \n\n"
            "Ambil foto ulang dengan objek utama terlihat jelas. "
            "Tidak ada skor yang diberikan."
        )

        return (
            gr.Dropdown(value=None),
            status,
            hasil,
        )

    if status_hasil == "perlu_verifikasi":
        terdeteksi = validasi.get(
            "komponen_terdeteksi",
            "tidak diketahui",
        )

        keyakinan = (
            validasi.get(
                "keyakinan_komponen_diminta",
                0,
            )
            * 100
        )

        status = (
            "⚠️ **Foto perlu diverifikasi petugas.**  \n"
            f"Hasil AI: **{hasil['nama']}** — skor **{hasil['skor']}**  \n"
            f"Komponen paling mirip: **{terdeteksi}** · "
            f"keyakinan foto {task}: **{keyakinan:.1f}%**  \n"
            f"Inferensi: {hasil['durasi_inferensi_detik']:.3f} detik · "
            f"Total: {hasil['durasi_total_detik']:.3f} detik"
        )

        return (
            gr.Dropdown(value=hasil["hasil"]),
            status,
            hasil,
        )

    status = (
        "✅ **Foto diterima.**  \n"
        f"Hasil AI: **{hasil['nama']}** — skor **{hasil['skor']}**  \n"
        f"Keyakinan relatif: "
        f"{hasil['keyakinan_relatif'] * 100:.1f}% · "
        f"Inferensi: "
        f"{hasil['durasi_inferensi_detik']:.3f} detik · "
        f"Total: "
        f"{hasil['durasi_total_detik']:.3f} detik"
    )

    return (
        gr.Dropdown(value=hasil["hasil"]),
        status,
        hasil,
    )


def analisis_atap(image):
    return analisis_clip_ui("atap", image)


def analisis_dinding(image):
    return analisis_clip_ui("dinding", image)


def analisis_lantai(image):
    return analisis_clip_ui("lantai", image)


def analisis_air(image):
    return analisis_clip_ui("air", image)


def analisis_wc(image):
    return analisis_clip_ui("wc", image)


def analisis_luas_ui(image_path, jumlah_anggota):
    if not image_path:
        raise gr.Error(
            "Ambil atau unggah foto tampak rumah terlebih dahulu."
        )

    if jumlah_anggota is None:
        raise gr.Error(
            "Jumlah anggota keluarga wajib diisi."
        )

    try:
        jumlah_anggota = int(jumlah_anggota)
    except (TypeError, ValueError) as error:
        raise gr.Error(
            "Jumlah anggota keluarga harus berupa angka."
        ) from error

    if jumlah_anggota < 1:
        raise gr.Error(
            "Jumlah anggota keluarga minimal 1 orang."
        )

    try:
        hasil = analisis(
            "luas",
            image_path,
            model=MODEL_DEFAULT,
        )

        hasil = tambahkan_skor(
            hasil,
            jumlah_anggota=jumlah_anggota,
        )
    except Exception as error:
        raise gr.Error(
            f"Analisis luas bangunan gagal: {error}"
        ) from error

    data = hasil["prediksi"]

    status = (
        f'✅ Perkiraan luas **{data["perkiraan_luas_m2"]:.1f} m²** · '
        f'**{data["luas_per_orang_m2"]:.1f} m²/orang** · '
        f'skor **{hasil["skor"]}** · '
        f'{hasil["durasi_detik"]:.2f} detik  \n\n'
        "> Estimasi visual, bukan pengukuran resmi. "
        "Petugas wajib memverifikasi."
    )

    return (
        data["perkiraan_luas_m2"],
        hasil["skor"],
        status,
        hasil,
    )


def hitung_total(
    skor_luas,
    atap,
    dinding,
    lantai,
    air,
    wc,
):
    nilai = {
        "luas": skor_luas,
        "atap": skor_pilihan("atap", atap),
        "dinding": skor_pilihan("dinding", dinding),
        "lantai": skor_pilihan("lantai", lantai),
        "air": skor_pilihan("air", air),
        "wc": skor_pilihan("wc", wc),
    }

    belum = [
        nama
        for nama, skor in nilai.items()
        if skor is None
    ]

    subtotal = sum(
        skor
        for skor in nilai.values()
        if skor is not None
    )

    if belum:
        return (
            f"## Skor sementara: {subtotal} / 18\n\n"
            f"Belum lengkap: {', '.join(belum)}"
        )

    return (
        f"## Total skor: {subtotal} / 18\n\n"
        "Semua komponen sudah dinilai. "
        "Hasil akhir tetap wajib diverifikasi petugas."
    )


def reset_hasil_foto():
    return (
        gr.Dropdown(value=None),
        "Foto berubah. Silakan lakukan analisis ulang.",
        None,
    )


def reset_hasil_luas():
    return (
        None,
        None,
        (
            "Foto atau jumlah anggota berubah. "
            "Silakan lakukan analisis ulang."
        ),
        None,
    )


def simpan_hasil(
    id_rumah,
    jumlah_anggota,
    luas_m2,
    skor_luas,
    atap,
    dinding,
    lantai,
    air,
    wc,
    hasil_luas,
    hasil_atap,
    hasil_dinding,
    hasil_lantai,
    hasil_air,
    hasil_wc,
):
    if not id_rumah or not id_rumah.strip():
        raise gr.Error("ID rumah wajib diisi.")

    if jumlah_anggota is None:
        raise gr.Error(
            "Jumlah anggota keluarga wajib diisi."
        )

    try:
        jumlah_anggota = int(jumlah_anggota)
    except (TypeError, ValueError) as error:
        raise gr.Error(
            "Jumlah anggota keluarga harus berupa angka."
        ) from error

    if jumlah_anggota < 1:
        raise gr.Error(
            "Jumlah anggota keluarga minimal 1 orang."
        )

    pilihan = {
        "atap": atap,
        "dinding": dinding,
        "lantai": lantai,
        "air": air,
        "wc": wc,
    }

    if skor_luas is None:
        raise gr.Error(
            "Luas bangunan belum dianalisis."
        )

    komponen_belum_lengkap = [
        task
        for task, value in pilihan.items()
        if not value
    ]

    if komponen_belum_lengkap:
        raise gr.Error(
            "Komponen berikut belum lengkap: "
            + ", ".join(komponen_belum_lengkap)
        )

    skor = {
        "luas": int(skor_luas),
    }

    skor.update(
        {
            task: skor_pilihan(task, value)
            for task, value in pilihan.items()
        }
    )

    if any(value is None for value in skor.values()):
        raise gr.Error(
            "Terdapat pilihan yang tidak sesuai "
            "dengan kategori penilaian."
        )

    hasil_ai = {
        "luas": hasil_luas,
        "atap": hasil_atap,
        "dinding": hasil_dinding,
        "lantai": hasil_lantai,
        "air": hasil_air,
        "wc": hasil_wc,
    }

    perlu_verifikasi = [
        task
        for task, hasil in hasil_ai.items()
        if isinstance(hasil, dict)
        and hasil.get("status") == "perlu_verifikasi"
    ]

    data = {
        "id_rumah": id_rumah.strip(),
        "tanggal": datetime.now().isoformat(
            timespec="seconds"
        ),
        "jumlah_anggota": jumlah_anggota,
        "perkiraan_luas_m2": luas_m2,
        "pilihan_petugas": pilihan,
        "skor": skor,
        "total": sum(skor.values()),
        "maksimum": 18,
        "komponen_perlu_verifikasi": perlu_verifikasi,
        "hasil_ai": hasil_ai,
        "model_klasifikasi": (
            f"OpenCLIP {MODEL_NAME} ({PRETRAINED})"
        ),
        "model_luas": MODEL_DEFAULT,
        "catatan": (
            "Hasil AI adalah observasi awal "
            "dan wajib diverifikasi petugas."
        ),
    }

    nama_aman = "".join(
        karakter
        if karakter.isalnum() or karakter in "-_"
        else "_"
        for karakter in id_rumah.strip()
    )

    path = OUTPUT_FOLDER / f"{nama_aman}_hasil.json"

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if perlu_verifikasi:
        status_simpan = (
            "⚠️ Hasil berhasil disimpan, tetapi komponen berikut "
            "masih memerlukan verifikasi petugas: "
            + ", ".join(perlu_verifikasi)
        )
    else:
        status_simpan = "✅ Hasil berhasil disimpan."

    return (
        data,
        str(path.resolve()),
        status_simpan,
    )


CSS = """
.gradio-container {
    max-width: 780px !important;
    margin: auto !important;
    background: #f2f1ed;
}

.header {
    background: #1175bd;
    color: white;
    padding: 22px;
    border-radius: 0 0 14px 14px;
    text-align: center;
}

.theme-button {
    max-width: 230px;
    margin: 12px 0 12px auto;
}

.step {
    text-align: center;
    color: #126dad;
    font-weight: 700;
    margin: 14px 0;
}

.card {
    background: white;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px #00000010;
}

.status {
    background: #eef7ff;
    border-left: 4px solid #1976d2;
    padding: 10px;
    border-radius: 6px;
}

.total {
    background: #e8f5e9;
    border: 1px solid #9bd1a1;
    padding: 18px;
    border-radius: 10px;
}

/* Mode gelap */
.dark .gradio-container {
    background: #111827 !important;
    color: #f3f4f6 !important;
}

.dark .card {
    background: #1f2937 !important;
    color: #f3f4f6 !important;
    box-shadow: 0 2px 10px #00000050;
}

.dark .status {
    background: #172554 !important;
    color: #dbeafe !important;
    border-left-color: #60a5fa;
}

.dark .total {
    background: #052e16 !important;
    color: #dcfce7 !important;
    border-color: #22c55e;
}

.dark .step {
    color: #60a5fa !important;
}

.dark input,
.dark textarea {
    color: #f3f4f6 !important;
}
"""


TOGGLE_THEME_JS = """
() => {
    const root = document.documentElement;
    const menjadiGelap = !root.classList.contains("dark");

    root.classList.toggle("dark", menjadiGelap);

    localStorage.setItem(
        "mode_tampilan",
        menjadiGelap ? "dark" : "light"
    );

    return menjadiGelap
        ? "☀️ Gunakan Mode Terang"
        : "🌙 Gunakan Mode Gelap";
}
"""


LOAD_THEME_JS = """
() => {
    const modeTersimpan = localStorage.getItem("mode_tampilan");

    if (modeTersimpan === "dark") {
        document.documentElement.classList.add("dark");
        return "☀️ Gunakan Mode Terang";
    }

    document.documentElement.classList.remove("dark");
    return "🌙 Gunakan Mode Gelap";
}
"""


with gr.Blocks(
    title="Data Prospek Subsidi - Kondisi Rumah"
) as app:
    gr.Markdown(
        "# Data Prospek Subsidi\n### Kondisi Rumah",
        elem_classes=["header"],
    )

    tombol_tema = gr.Button(
        "🌙 Gunakan Mode Gelap",
        elem_classes=["theme-button"],
    )

    gr.Markdown(
        "① Informasi Nasabah　"
        "**② Kondisi Rumah**　"
        "③ Tingkat Pendapatan　"
        "④ Sektor Ekonomi",
        elem_classes=["step"],
    )

    id_rumah = gr.Textbox(
        label="ID rumah *",
        placeholder="Contoh: RUMAH-001",
    )

    jumlah_anggota = gr.Number(
        label="Jumlah anggota keluarga *",
        minimum=1,
        precision=0,
        value=1,
    )

    with gr.Group(elem_classes=["card"]):
        gr.Markdown("## Luas Bangunan")

        luas_m2 = gr.Number(
            label="Perkiraan luas AI (m²)",
            interactive=False,
        )

        foto_luas = gr.Image(
            label="Foto Luas Bangunan *",
            type="filepath",
            sources=["upload", "webcam"],
        )

        tombol_luas = gr.Button(
            "📷 Ambil / Analisis Foto Luas Bangunan",
            variant="primary",
        )

        status_luas = gr.Markdown(
            "Belum dianalisis.",
            elem_classes=["status"],
        )

    komponen = {}

    judul = {
        "atap": "Jenis Atap",
        "dinding": "Dinding",
        "lantai": "Lantai",
        "air": "Sumber Air",
        "wc": "WC",
    }

    for task in TASKS:
        with gr.Group(elem_classes=["card"]):
            gr.Markdown(f"## {judul[task]}")

            dropdown = gr.Dropdown(
                label=f"{judul[task]} *",
                choices=pilihan_dropdown(task),
                value=None,
                interactive=True,
                info=(
                    "Diisi otomatis oleh AI "
                    "dan dapat dikoreksi petugas."
                ),
            )

            image = gr.Image(
                label=f"Foto {judul[task]} *",
                type="filepath",
                sources=["upload", "webcam"],
            )

            button = gr.Button(
                f"📷 Ambil / Analisis Foto {judul[task]}",
                variant="primary",
            )

            status = gr.Markdown(
                "Belum dianalisis.",
                elem_classes=["status"],
            )

            state = gr.State(None)

            komponen[task] = (
                dropdown,
                image,
                button,
                status,
                state,
            )

    skor_luas_state = gr.State(None)
    hasil_luas_state = gr.State(None)

    total = gr.Markdown(
        "## Skor sementara: 0 / 18",
        elem_classes=["total"],
    )

    simpan = gr.Button(
        "Simpan Hasil Kondisi Rumah",
        variant="primary",
        size="lg",
    )

    status_simpan = gr.Markdown("")

    hasil_json = gr.JSON(
        label="Hasil lengkap"
    )

    file_json = gr.File(
        label="Unduh JSON"
    )

    # Tema terang/gelap
    tombol_tema.click(
        fn=None,
        inputs=None,
        outputs=tombol_tema,
        js=TOGGLE_THEME_JS,
        queue=False,
    )

    app.load(
        fn=None,
        inputs=None,
        outputs=tombol_tema,
        js=LOAD_THEME_JS,
        queue=False,
    )

    # Analisis luas bangunan
    tombol_luas.click(
        analisis_luas_ui,
        inputs=[
            foto_luas,
            jumlah_anggota,
        ],
        outputs=[
            luas_m2,
            skor_luas_state,
            status_luas,
            hasil_luas_state,
        ],
    )

    foto_luas.change(
        reset_hasil_luas,
        outputs=[
            luas_m2,
            skor_luas_state,
            status_luas,
            hasil_luas_state,
        ],
    )

    jumlah_anggota.change(
        reset_hasil_luas,
        outputs=[
            luas_m2,
            skor_luas_state,
            status_luas,
            hasil_luas_state,
        ],
    )

    fungsi = {
        "atap": analisis_atap,
        "dinding": analisis_dinding,
        "lantai": analisis_lantai,
        "air": analisis_air,
        "wc": analisis_wc,
    }

    # Analisis dan reset setiap komponen
    for task in TASKS:
        dropdown, image, button, status, state = komponen[task]

        button.click(
            fungsi[task],
            inputs=image,
            outputs=[
                dropdown,
                status,
                state,
            ],
        )

        image.change(
            reset_hasil_foto,
            outputs=[
                dropdown,
                status,
                state,
            ],
        )

    # Perhitungan total
    input_total = [
        skor_luas_state,
        *[komponen[task][0] for task in TASKS],
    ]

    skor_luas_state.change(
        hitung_total,
        inputs=input_total,
        outputs=total,
    )

    for task in TASKS:
        komponen[task][0].change(
            hitung_total,
            inputs=input_total,
            outputs=total,
        )

    # Simpan hasil
    simpan.click(
        simpan_hasil,
        inputs=[
            id_rumah,
            jumlah_anggota,
            luas_m2,
            skor_luas_state,
            *[komponen[task][0] for task in TASKS],
            hasil_luas_state,
            *[komponen[task][4] for task in TASKS],
        ],
        outputs=[
            hasil_json,
            file_json,
            status_simpan,
        ],
    )


if __name__ == "__main__":
    app.queue(
        max_size=10,
        default_concurrency_limit=1,
    )

    app.launch(
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="blue"
        ),
        css=CSS,
    )