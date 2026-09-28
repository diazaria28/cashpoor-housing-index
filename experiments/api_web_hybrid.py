"""REST API dan static server untuk UI JavaScript observasi rumah."""

import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_ROOT = Path(__file__).resolve().parent
for folder in (ROOT, EXPERIMENTS_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from analyzer import MODEL_DEFAULT, analisis, tambahkan_skor
from clip_analyzer import KATEGORI, MODEL_NAME, PRETRAINED, klasifikasi
from dinding_hybrid.dinding_hybrid import analisis_dinding as analisis_dinding_hybrid


WEB_ROOT = ROOT / "web_ui_hybrid"
OUTPUT_ROOT = ROOT / "hasil_web_hybrid"
OUTPUT_ROOT.mkdir(exist_ok=True)

app = FastAPI(title="Observasi Kondisi Rumah Hybrid API", version="2.0.0")


def simpan_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "foto.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(400, "Format gambar tidak didukung.")
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with handle:
        shutil.copyfileobj(upload.file, handle)
    return Path(handle.name)


@app.get("/api/config")
def config():
    return {
        "model": f"OpenCLIP {MODEL_NAME} ({PRETRAINED}) + Grounding DINO khusus campuran dinding",
        "model_luas": MODEL_DEFAULT,
        "maksimum_skor": 18,
        "kategori": {
            task: [
                {
                    "value": key,
                    "nama": data["nama"],
                    "skor": data["skor"],
                }
                for key, data in kategori.items()
            ]
            for task, kategori in KATEGORI.items()
        },
    }


@app.post("/api/analyze/component/{task}")
def analyze_component(task: str, image: UploadFile = File(...)):
    if task not in KATEGORI:
        raise HTTPException(404, "Komponen tidak dikenal.")
    path = simpan_upload(image)
    try:
        if task == "dinding":
            hasil = analisis_dinding_hybrid(str(path))
            hasil_clip = hasil.get("hasil_openclip") or {}
            # Bidang kompatibilitas agar UI JavaScript dapat menangani hasil hybrid.
            hasil["validasi_foto"] = hasil_clip.get("validasi_foto") or {}
            hasil["keyakinan_relatif"] = hasil.get("keyakinan_clip")
            return hasil
        return klasifikasi(task, str(path))
    except Exception as exc:
        raise HTTPException(500, f"Analisis gagal: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/analyze/luas")
def analyze_area(
    image: UploadFile = File(...),
    members: int = Form(...),
):
    if members < 1:
        raise HTTPException(400, "Jumlah anggota minimal satu orang.")
    path = simpan_upload(image)
    try:
        hasil = analisis("luas", str(path), model=MODEL_DEFAULT)
        return tambahkan_skor(hasil, jumlah_anggota=members)
    except Exception as exc:
        raise HTTPException(500, f"Analisis luas gagal: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/save")
def save_result(data: dict):
    house_id = str(data.get("id_rumah", "")).strip()
    if not house_id:
        raise HTTPException(400, "ID rumah wajib diisi.")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in house_id)
    data["tanggal"] = datetime.now().isoformat(timespec="seconds")
    data["model_klasifikasi"] = (
        f"OpenCLIP {MODEL_NAME} ({PRETRAINED}) + "
        "Grounding DINO khusus konfirmasi setengah tembok"
    )
    data["model_luas"] = MODEL_DEFAULT
    data["catatan"] = "Hasil AI adalah observasi awal dan wajib diverifikasi petugas."
    output = OUTPUT_ROOT / f"{safe_name}_hasil.json"
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "filename": output.name, "data": data}


@app.get("/api/download/{filename}")
def download(filename: str):
    safe = Path(filename).name
    path = OUTPUT_ROOT / safe
    if not path.is_file():
        raise HTTPException(404, "File tidak ditemukan.")
    return FileResponse(path, filename=safe, media_type="application/json")


app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")


@app.get("/")
def index():
    return FileResponse(WEB_ROOT / "index.html")
