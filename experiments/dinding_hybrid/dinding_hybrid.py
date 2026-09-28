"""Hybrid OpenCLIP + Grounding DINO khusus klasifikasi material dinding."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_clip = None
_gdino = None


def muat_modul():
    global _clip, _gdino
    if _clip is None:
        _clip = _load("clip_baseline", ROOT / "clip_analyzer.py")
    if _gdino is None:
        _gdino = _load(
            "dinding_grounding_dino",
            ROOT / "experiments" / "dinding_grounding_dino" / "dinding_grounding_dino.py",
        )
    return _clip, _gdino


def statistik_clip(hasil_clip):
    probs = hasil_clip.get("semua_probabilitas") or {}
    urut = sorted((float(v), k) for k, v in probs.items())
    confidence = urut[-1][0] if urut else 0.0
    margin = confidence - urut[-2][0] if len(urut) > 1 else confidence
    return confidence, margin


def klasifikasi_openclip_dinding_toleran(image_path):
    """Longgarkan hanya konflik dinding-vs-lantai pada foto dominan dinding."""
    clip, _ = muat_modul()
    hasil = clip.klasifikasi("dinding", image_path)
    validasi = hasil.get("validasi_foto") or {}
    if not (
        hasil.get("status") == "foto_tidak_sesuai"
        and validasi.get("komponen_terdeteksi") == "lantai"
    ):
        return hasil

    prompts = {
        "dinding_dominan": [
            "a large plain painted interior wall occupying most of the image with only a narrow strip of floor",
            "a smooth vertical plastered wall photographed straight from the front",
            "an empty painted masonry wall with a baseboard and a small visible floor edge",
            "a vertical house wall surface dominating the photograph",
        ],
        "lantai_dominan": [
            "a photograph focused downward on an indoor floor occupying most of the image",
            "a wooden plank floor surface filling most of the photograph",
            "a room floor photographed from above with only a small part of the wall",
            "a horizontal floor surface as the main subject of the image",
        ],
    }
    with Image.open(image_path) as gambar:
        tensor = clip._preprocess(gambar.convert("RGB")).unsqueeze(0).to(clip.DEVICE)
    with torch.inference_mode():
        image_features = clip.normalisasi_fitur(clip._model.encode_image(tensor))
        prototypes = []
        for daftar_prompt in prompts.values():
            tokens = clip._tokenizer(daftar_prompt).to(clip.DEVICE)
            features = clip.normalisasi_fitur(clip._model.encode_text(tokens))
            prototypes.append(clip.normalisasi_fitur(features.mean(dim=0, keepdim=True))[0])
        probs_geometri = (100.0 * image_features @ torch.stack(prototypes).T).softmax(dim=-1)[0]
        prob_dinding_dominan = float(probs_geometri[0].item())

    # Foto lantai murni tetap ditolak; toleransi hanya untuk komposisi dominan vertikal.
    if prob_dinding_dominan < 0.65:
        hasil["validasi_dinding_toleran"] = {
            "diterapkan": True,
            "lolos": False,
            "probabilitas_dinding_dominan": round(prob_dinding_dominan, 4),
        }
        return hasil

    labels, text_features = clip.fitur_teks("dinding")
    with torch.inference_mode():
        probabilities = (clip.TEMPERATURE_KATEGORI * image_features @ text_features.T).softmax(dim=-1)[0]
    index = int(probabilities.argmax().item())
    label = labels[index]
    data = clip.KATEGORI["dinding"][label]
    hasil.update({
        "status": "perlu_verifikasi",
        "hasil": label,
        "nama": data["nama"],
        "skor": data["skor"],
        "keyakinan_relatif": round(float(probabilities[index].item()), 4),
        "semua_probabilitas": {
            labels[i]: round(float(probabilities[i].item()), 4) for i in range(len(labels))
        },
        "validasi_dinding_toleran": {
            "diterapkan": True,
            "lolos": True,
            "probabilitas_dinding_dominan": round(prob_dinding_dominan, 4),
            "alasan": "Bidang dinding dominan; lantai hanya konteks di bagian bawah.",
        },
    })
    return hasil


def perlu_grounding(hasil_clip):
    if hasil_clip.get("status") != "berhasil":
        return False
    return hasil_clip.get("hasil") in {"setengah_tembok", "kayu_bambu"}


def grounding_mengonfirmasi_campuran(hasil_grounding):
    if not hasil_grounding or hasil_grounding.get("status") != "berhasil":
        return False
    if hasil_grounding.get("hasil") != "setengah_tembok":
        return False
    proporsi = hasil_grounding.get("proporsi_material") or {}
    confidence = hasil_grounding.get("confidence_per_kelompok") or {}
    cakupan = float(hasil_grounding.get("cakupan_gambar", 0.0))
    return (
        float(proporsi.get("tembok", 0.0)) >= 0.20
        and float(proporsi.get("kayu_bambu", 0.0)) >= 0.20
        and float(confidence.get("tembok", 0.0)) >= 0.24
        and float(confidence.get("kayu_bambu", 0.0)) >= 0.24
        and cakupan >= 0.40
    )


def gabungkan_hasil(hasil_clip, hasil_grounding=None):
    """Aturan fusi yang dapat diuji tanpa memuat ulang kedua model."""
    clip_label = hasil_clip.get("hasil")
    clip_conf, clip_margin = statistik_clip(hasil_clip)
    dipanggil = hasil_grounding is not None

    if not dipanggil:
        final = clip_label
        status = hasil_clip.get("status", "berhasil")
        alasan = "OpenCLIP mendeteksi tembok dengan keyakinan dan margin yang kuat."
    elif hasil_grounding.get("status") != "berhasil" or not hasil_grounding.get("hasil"):
        final = clip_label
        status = hasil_clip.get("status", "berhasil")
        alasan = "Grounding DINO tidak memberi bukti campuran kuat; hasil dan status OpenCLIP dipertahankan."
    else:
        gd_label = hasil_grounding["hasil"]
        if gd_label == clip_label:
            final, status = clip_label, "berhasil"
            alasan = "OpenCLIP dan Grounding DINO sepakat."
        elif grounding_mengonfirmasi_campuran(hasil_grounding):
            final, status = gd_label, "berhasil"
            alasan = "Grounding DINO menemukan dua material dengan cakupan dan confidence memadai."
        else:
            final = clip_label
            status = hasil_clip.get("status", "berhasil")
            alasan = "Bukti campuran Grounding DINO belum memenuhi ambang; hasil OpenCLIP dipertahankan."

    skor = {"tembok": 3, "setengah_tembok": 1, "kayu_bambu": 0}.get(final)
    nama = {
        "tembok": "Tembok penuh / bata / batako",
        "setengah_tembok": "Setengah tembok / campuran",
        "kayu_bambu": "Kayu biasa / bambu / rumbia",
    }.get(final)
    return {
        "status": status,
        "hasil": final,
        "nama": nama,
        "skor": skor,
        "grounding_dino_dipanggil": dipanggil,
        "alasan_fusi": alasan,
        "keyakinan_clip": round(clip_conf, 4),
        "margin_clip": round(clip_margin, 4),
        "hasil_openclip": hasil_clip,
        "hasil_grounding_dino": hasil_grounding,
    }


def analisis_dinding(image_path):
    clip, gdino = muat_modul()
    awal = time.perf_counter()
    hasil_clip = klasifikasi_openclip_dinding_toleran(image_path)
    hasil_grounding = gdino.analisis_dinding(image_path) if perlu_grounding(hasil_clip) else None
    output = gabungkan_hasil(hasil_clip, hasil_grounding)
    output["durasi_total_detik"] = round(time.perf_counter() - awal, 4)
    return output


def main():
    parser = argparse.ArgumentParser(description="Hybrid OpenCLIP + Grounding DINO untuk dinding")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(json.dumps(analisis_dinding(args.image), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
