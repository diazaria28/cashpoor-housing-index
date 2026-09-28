"""Eksperimen zero-shot deteksi material dinding dengan Grounding DINO."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


MODEL_NAME = "IDEA-Research/grounding-dino-tiny"
PROMPTS = [
    "plastered masonry wall",
    "painted stucco exterior wall",
    "painted concrete house wall",
    "brick wall",
    "wooden plank wall",
    "woven bamboo wall",
]
GROUP_MAPPING = {
    "plastered masonry wall": "tembok",
    "painted stucco exterior wall": "tembok",
    "painted concrete house wall": "tembok",
    "brick wall": "tembok",
    "wooden plank wall": "kayu_bambu",
    "woven bamboo wall": "kayu_bambu",
}
NAMA = {
    "tembok": "Tembok",
    "setengah_tembok": "Setengah tembok / campuran",
    "kayu_bambu": "Kayu / bambu",
}
SKOR = {"tembok": 3, "setengah_tembok": 1, "kayu_bambu": 0}
_processor = None
_model = None
_device = None


def muat_model():
    global _processor, _model, _device
    if _model is None:
        print("Memuat Grounding DINO Tiny...", flush=True)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)
        # Resolusi lebih kecil membuat evaluasi CPU jauh lebih praktis.
        _processor.image_processor.size = {"shortest_edge": 480, "longest_edge": 640}
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_NAME)
        _model.to(_device).eval()
        print(f"Grounding DINO siap ({_device}).", flush=True)
    return _processor, _model, _device


def _tidak_yakin(alasan, durasi, deteksi=None, proporsi=None, kandidat=None):
    return {
        "status": "perlu_verifikasi",
        "hasil": kandidat,
        "nama": NAMA.get(kandidat),
        "skor": SKOR.get(kandidat),
        "alasan_verifikasi": alasan,
        "proporsi_material": proporsi or {"tembok": 0.0, "kayu_bambu": 0.0},
        "durasi_detik": round(durasi, 4),
        "deteksi": deteksi or [],
    }


def analisis_dinding(image_path, box_threshold=0.20, text_threshold=0.20, grid_size=256):
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Foto tidak ditemukan: {path}")

    processor, model, device = muat_model()
    image = Image.open(path).convert("RGB")
    text = ". ".join(PROMPTS) + "."
    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    awal = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]
    if device == "cuda":
        torch.cuda.synchronize()
    durasi = time.perf_counter() - awal

    skor_grid = {
        "tembok": np.zeros((grid_size, grid_size), dtype=np.float32),
        "kayu_bambu": np.zeros((grid_size, grid_size), dtype=np.float32),
    }
    deteksi = []
    labels = results.get("text_labels", results.get("labels", []))
    for box, score, raw_label in zip(results["boxes"], results["scores"], labels):
        confidence = float(score.item())
        label = str(raw_label).strip().lower().rstrip(".")
        group = GROUP_MAPPING.get(label)
        if group is None:
            ada_tembok = any(k in label for k in ("masonry", "stucco", "concrete", "brick"))
            ada_kayu = any(k in label for k in ("wood", "wooden", "bamboo"))
            # Frasa gabungan berasal dari token yang ambigu, bukan bukti dua material.
            group = "tembok" if ada_tembok and not ada_kayu else (
                "kayu_bambu" if ada_kayu and not ada_tembok else None
            )
        if group is None:
            continue

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        gx1 = max(0, min(grid_size - 1, int(x1 / image.width * grid_size)))
        gy1 = max(0, min(grid_size - 1, int(y1 / image.height * grid_size)))
        gx2 = max(gx1 + 1, min(grid_size, int(np.ceil(x2 / image.width * grid_size))))
        gy2 = max(gy1 + 1, min(grid_size, int(np.ceil(y2 / image.height * grid_size))))
        skor_grid[group][gy1:gy2, gx1:gx2] = np.maximum(
            skor_grid[group][gy1:gy2, gx1:gx2], confidence
        )
        deteksi.append({
            "prompt": label,
            "kelompok": group,
            "confidence": round(confidence, 4),
            "rasio_luas_box": round(max(0.0, (x2 - x1) * (y2 - y1)) / (image.width * image.height), 4),
            "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        })

    if not deteksi:
        return _tidak_yakin(["Grounding DINO tidak menemukan material dinding."], durasi)

    gabungan = np.stack([skor_grid["tembok"], skor_grid["kayu_bambu"]])
    aktif = gabungan.max(axis=0) > 0
    cakupan = float(aktif.mean())
    pemilik = gabungan.argmax(axis=0)
    jumlah_tembok = int(((pemilik == 0) & aktif).sum())
    jumlah_kayu = int(((pemilik == 1) & aktif).sum())
    total = jumlah_tembok + jumlah_kayu
    prop_tembok = jumlah_tembok / total
    prop_kayu = jumlah_kayu / total
    proporsi = {"tembok": round(prop_tembok, 4), "kayu_bambu": round(prop_kayu, 4)}
    conf_group = {
        group: max((d["confidence"] for d in deteksi if d["kelompok"] == group), default=0.0)
        for group in ("tembok", "kayu_bambu")
    }

    if prop_tembok >= 0.75:
        hasil = "tembok"
    elif prop_kayu >= 0.75:
        hasil = "kayu_bambu"
    elif prop_tembok >= 0.20 and prop_kayu >= 0.20:
        hasil = "setengah_tembok"
    else:
        hasil = "tembok" if prop_tembok > prop_kayu else "kayu_bambu"

    alasan = []
    if hasil == "tembok" and (cakupan < 0.30 or conf_group["tembok"] < 0.15):
        alasan.append("Bukti kotak tembok belum cukup kuat atau luas.")
    elif hasil == "kayu_bambu" and (cakupan < 0.25 or conf_group["kayu_bambu"] < 0.18):
        alasan.append("Bukti kotak kayu/bambu belum cukup kuat atau luas.")
    elif hasil == "setengah_tembok" and (
        cakupan < 0.30 or min(conf_group.values()) < 0.12
    ):
        alasan.append("Kedua material belum terdeteksi cukup kuat untuk kelas campuran.")

    if alasan:
        output = _tidak_yakin(alasan, durasi, deteksi, proporsi, hasil)
    else:
        output = {
            "status": "berhasil",
            "hasil": hasil,
            "nama": NAMA[hasil],
            "skor": SKOR[hasil],
            "proporsi_material": proporsi,
            "durasi_detik": round(durasi, 4),
            "deteksi": deteksi,
        }
    output["cakupan_gambar"] = round(cakupan, 4)
    output["confidence_per_kelompok"] = {k: round(v, 4) for k, v in conf_group.items()}
    return output


def main():
    parser = argparse.ArgumentParser(description="Eksperimen Grounding DINO untuk material dinding")
    parser.add_argument("--image", required=True)
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    args = parser.parse_args()
    hasil = analisis_dinding(args.image, args.box_threshold, args.text_threshold)
    print(json.dumps(hasil, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
