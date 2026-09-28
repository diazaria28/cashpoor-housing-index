import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

DETAIL_PATH = (
    PROJECT_ROOT
    / "hasil_evaluasi_dataset"
    / "detail.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "hasil_evaluasi_dataset"
    / "confusion_matrix"
)


URUTAN_KELAS = {
    "atap": [
        "atap_bagus",
        "atap_biasa",
        "atap_daun",
    ],
    "dinding": [
        "tembok",
        "setengah_tembok",
        "kayu_bambu",
    ],
    "lantai": [
        "keramik",
        "semen",
        "tanah_kayu",
    ],
    "air": [
        "air_sangat_layak",
        "air_layak",
        "air_tidak_layak",
    ],
    "wc": [
        "wc_duduk",
        "wc_jongkok",
        "wc_lainnya",
    ],
}


NAMA_TAMPILAN = {
    "atap_bagus": "Atap bagus",
    "atap_biasa": "Atap biasa",
    "atap_daun": "Atap daun",

    "tembok": "Tembok",
    "setengah_tembok": "Setengah tembok",
    "kayu_bambu": "Kayu/bambu",

    "keramik": "Keramik",
    "semen": "Semen",
    "tanah_kayu": "Tanah/kayu",

    "air_sangat_layak": "Sangat layak",
    "air_layak": "Layak",
    "air_tidak_layak": "Tidak layak",

    "wc_duduk": "WC duduk",
    "wc_jongkok": "WC jongkok",
    "wc_lainnya": "WC lainnya",

    "ditolak": "Ditolak",
}


def baca_detail():
    if not DETAIL_PATH.is_file():
        raise FileNotFoundError(
            f"File tidak ditemukan: {DETAIL_PATH}\n"
            "Jalankan evaluasi dataset terlebih dahulu."
        )

    with DETAIL_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def nilai_boolean(value):
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def buat_matrix(rows, task, kelas):
    """
    Baris   = label acuan
    Kolom   = prediksi model

    Kolom 'ditolak' digunakan untuk foto valid yang tidak
    menghasilkan prediksi.
    """

    kelas_prediksi = kelas + ["ditolak"]

    matrix = np.zeros(
        (
            len(kelas),
            len(kelas_prediksi),
        ),
        dtype=int,
    )

    index_acuan = {
        label: index
        for index, label in enumerate(kelas)
    }

    index_prediksi = {
        label: index
        for index, label in enumerate(kelas_prediksi)
    }

    for row in rows:
        if row["task"] != task:
            continue

        if not nilai_boolean(row["foto_sesuai"]):
            continue

        label_acuan = row["label_acuan"].strip()
        prediksi = row["prediksi"].strip()

        if not prediksi:
            prediksi = "ditolak"

        if label_acuan not in index_acuan:
            print(
                f"Peringatan: label acuan tidak dikenal: "
                f"{label_acuan}"
            )
            continue

        if prediksi not in index_prediksi:
            print(
                f"Peringatan: prediksi tidak dikenal: "
                f"{prediksi}"
            )
            prediksi = "ditolak"

        matrix[
            index_acuan[label_acuan],
            index_prediksi[prediksi],
        ] += 1

    return matrix, kelas_prediksi


def normalisasi_matrix(matrix):
    jumlah_baris = matrix.sum(
        axis=1,
        keepdims=True,
    )

    return np.divide(
        matrix,
        jumlah_baris,
        out=np.zeros_like(
            matrix,
            dtype=float,
        ),
        where=jumlah_baris != 0,
    )


def simpan_csv_matrix(
    path,
    matrix,
    label_baris,
    label_kolom,
    normalized=False,
):
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            ["label_acuan"] + label_kolom
        )

        for index, label in enumerate(label_baris):
            nilai = matrix[index]

            if normalized:
                nilai = [
                    round(float(item), 4)
                    for item in nilai
                ]
            else:
                nilai = [
                    int(item)
                    for item in nilai
                ]

            writer.writerow(
                [label] + nilai
            )


def gambar_matrix(
    ax,
    matrix,
    task,
    label_baris,
    label_kolom,
):
    normalized = normalisasi_matrix(matrix)

    image = ax.imshow(
        normalized,
        cmap="Blues",
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    ax.set_title(
        (
            f"{task.upper()}\n"
            f"Jumlah data: {matrix.sum()}"
        ),
        fontsize=12,
        fontweight="bold",
    )

    ax.set_xlabel("Prediksi model")
    ax.set_ylabel("Label acuan")

    ax.set_xticks(
        range(len(label_kolom))
    )

    ax.set_yticks(
        range(len(label_baris))
    )

    ax.set_xticklabels(
        [
            NAMA_TAMPILAN.get(label, label)
            for label in label_kolom
        ],
        rotation=35,
        ha="right",
        fontsize=9,
    )

    ax.set_yticklabels(
        [
            NAMA_TAMPILAN.get(label, label)
            for label in label_baris
        ],
        fontsize=9,
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            jumlah = matrix[
                row_index,
                column_index,
            ]

            persen = normalized[
                row_index,
                column_index,
            ] * 100

            warna = (
                "white"
                if normalized[
                    row_index,
                    column_index
                ] >= 0.5
                else "black"
            )

            ax.text(
                column_index,
                row_index,
                f"{jumlah}\n({persen:.1f}%)",
                ha="center",
                va="center",
                color=warna,
                fontsize=9,
                fontweight=(
                    "bold"
                    if jumlah > 0
                    else "normal"
                ),
            )

    return image


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = baca_detail()

    # Figure gabungan lima variabel.
    figure, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(15, 17),
    )

    axes = axes.flatten()

    image_terakhir = None

    for index, (task, kelas) in enumerate(
        URUTAN_KELAS.items()
    ):
        matrix, kelas_prediksi = buat_matrix(
            rows,
            task,
            kelas,
        )

        # Menyimpan matriks jumlah mentah.
        simpan_csv_matrix(
            OUTPUT_DIR
            / f"confusion_matrix_{task}.csv",
            matrix,
            kelas,
            kelas_prediksi,
            normalized=False,
        )

        # Menyimpan matriks yang sudah dinormalisasi.
        normalized = normalisasi_matrix(matrix)

        simpan_csv_matrix(
            OUTPUT_DIR
            / f"confusion_matrix_{task}_normalized.csv",
            normalized,
            kelas,
            kelas_prediksi,
            normalized=True,
        )

        image_terakhir = gambar_matrix(
            axes[index],
            matrix,
            task,
            kelas,
            kelas_prediksi,
        )

    # Menghapus subplot kosong.
    for index in range(
        len(URUTAN_KELAS),
        len(axes),
    ):
        figure.delaxes(axes[index])

    figure.suptitle(
        "Confusion Matrix Klasifikasi Kondisi Rumah",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    figure.text(
        0.5,
        0.006,
        (
            "Nilai dalam kotak menunjukkan jumlah foto "
            "dan persentase terhadap label acuan."
        ),
        ha="center",
        fontsize=10,
    )

    figure.tight_layout(
        rect=[0, 0.02, 1, 0.98]
    )

    output_png = (
        OUTPUT_DIR
        / "confusion_matrix_semua_variabel.png"
    )

    figure.savefig(
        output_png,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Confusion matrix berhasil dibuat.")
    print(f"Folder hasil: {OUTPUT_DIR.resolve()}")
    print(f"Gambar gabungan: {output_png.resolve()}")


if __name__ == "__main__":
    main()