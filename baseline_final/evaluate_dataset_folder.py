import json
from pathlib import Path

from evaluasi_clip import evaluasi, ringkas, simpan_csv


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "hasil_evaluasi_dataset"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def buat_manifest():
    """
    Membaca dataset dengan struktur:

    dataset/
    ├── atap/
    │   ├── atap_bagus/
    │   ├── atap_biasa/
    │   └── atap_daun/
    ├── dinding/
    ├── lantai/
    ├── air/
    ├── wc/
    └── salah_kolom/
    """

    manifest = []

    # =========================================================
    # 1. FOTO YANG BERADA PADA KOLOM YANG BENAR
    # =========================================================

    for task_dir in sorted(DATASET_ROOT.iterdir()):
        if not task_dir.is_dir():
            continue

        if task_dir.name == "salah_kolom":
            continue

        task = task_dir.name

        for label_dir in sorted(task_dir.iterdir()):
            if not label_dir.is_dir():
                continue

            label = label_dir.name

            for image_path in sorted(label_dir.rglob("*")):
                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                manifest.append(
                    {
                        "id": (
                            f"{task}_"
                            f"{label}_"
                            f"{image_path.stem}"
                        ),
                        "task": task,
                        "image": str(image_path.resolve()),
                        "foto_sesuai": True,
                        "label_acuan": label,
                    }
                )

    # =========================================================
    # 2. FOTO SALAH KOLOM
    # =========================================================
    #
    # Struktur yang didukung:
    #
    # dataset/
    # └── salah_kolom/
    #     ├── atap/
    #     │   ├── foto_dinding.jpg
    #     │   └── foto_lantai.jpg
    #     ├── dinding/
    #     ├── lantai/
    #     ├── air/
    #     └── wc/
    #
    # Nama subfolder menunjukkan kolom tempat foto diuji.
    # Contoh:
    # salah_kolom/atap/foto_dinding.jpg
    # berarti foto dinding sengaja dimasukkan ke kolom atap.

    salah_kolom_root = DATASET_ROOT / "salah_kolom"

    if salah_kolom_root.is_dir():
        for task_dir in sorted(salah_kolom_root.iterdir()):
            if not task_dir.is_dir():
                continue

            task = task_dir.name

            for image_path in sorted(task_dir.rglob("*")):
                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                manifest.append(
                    {
                        "id": (
                            f"salah_{task}_"
                            f"{image_path.stem}"
                        ),
                        "task": task,
                        "image": str(image_path.resolve()),
                        "foto_sesuai": False,
                        "label_acuan": None,
                    }
                )

    return manifest


def tampilkan_jumlah_per_kategori(manifest):
    jumlah = {}

    for item in manifest:
        if item["foto_sesuai"]:
            key = (
                item["task"],
                item["label_acuan"],
            )
        else:
            key = (
                "salah_kolom",
                item["task"],
            )

        jumlah[key] = jumlah.get(key, 0) + 1

    print("\nJumlah data per kategori")
    print("=" * 60)

    for key, total in sorted(jumlah.items()):
        nama = " / ".join(key)
        print(f"{nama:<45} : {total}")

    print("=" * 60)
    print(f"Total gambar: {len(manifest)}\n")


def main():
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {DATASET_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Membuat manifest otomatis dari struktur folder.
    manifest = buat_manifest()

    if not manifest:
        raise RuntimeError(
            f"Tidak ada gambar yang ditemukan dalam {DATASET_ROOT}"
        )

    tampilkan_jumlah_per_kategori(manifest)

    # Menyimpan manifest agar susunan data yang diuji
    # dapat diperiksa kembali.
    manifest_path = OUTPUT_ROOT / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Mulai evaluasi model...\n")

    # Menjalankan model untuk seluruh gambar.
    detail = evaluasi(manifest)

    # Menghitung ringkasan metrik.
    ringkasan = ringkas(detail)

    # Menyimpan detail hasil setiap gambar.
    detail_path = OUTPUT_ROOT / "detail.csv"

    simpan_csv(
        detail_path,
        detail,
    )

    # Menyimpan ringkasan metrik.
    ringkasan_path = OUTPUT_ROOT / "ringkasan.json"

    ringkasan_path.write_text(
        json.dumps(
            ringkasan,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n")
    print("=" * 60)
    print("RINGKASAN HASIL EVALUASI")
    print("=" * 60)

    print(
        json.dumps(
            ringkasan,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nFile hasil:")
    print(f"Manifest  : {manifest_path.resolve()}")
    print(f"Detail    : {detail_path.resolve()}")
    print(f"Ringkasan : {ringkasan_path.resolve()}")


if __name__ == "__main__":
    main()