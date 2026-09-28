import argparse
import json
import time
from pathlib import Path

import open_clip
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "dataset"

def resolve_image_path(image_path):
    path = Path(image_path).expanduser()

    candidates = [
        path,                  # path absolut / relatif dari terminal
        ROOT_DIR / path,       # relatif dari folder proyek
        DATA_DIR / path,       # relatif dari folder dataset
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Foto tidak ditemukan: {image_path}"
    )



MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEMPERATURE_KATEGORI = 100.0


KATEGORI = {
    "atap": {
        "atap_bagus": {
            "skor": 3,
            "nama": "Genteng bagus / beton / multiroof",
            "prompts": [
                "a house roof made of high quality concrete roof tiles",
                "a modern multiroof metal tile house roof",
                "a solid concrete roof of a house",
            ],
        },
        "atap_biasa": {
            "skor": 1,
            "nama": "Genteng tanah liat / seng / asbes",
            "prompts": [
                "an Indonesian house roof made of corrugated metal sheets",
                "a corrugated fiber cement asbestos house roof seen from below",
                "an ordinary clay tile roof of a modest Indonesian house",
            ],
        },
        "atap_daun": {
            "skor": 0,
            "nama": "Rumbia / anyaman daun",
            "prompts": [
                "a traditional thatched roof made of palm leaves",
                "a woven dried leaf roof on a simple house",
            ],
        },
    },

    "dinding": {
        "tembok": {
            "skor": 3,
            "nama": "Tembok penuh / bata / batako",
            "prompts": [
                "a close-up photo of a house wall entirely made of plastered brick masonry",
                "a permanent house wall completely made of concrete blocks",
                "a full solid cement rendered residential wall",
                "a painted masonry house wall without bamboo or wooden wall panels",
                "a house facade dominated entirely by brick concrete or plastered masonry",
                "a ceramic tiled wall with solid brick masonry underneath",
            ],
        },

        "setengah_tembok": {
            "skor": 1,
            "nama": "Setengah tembok / campuran",
            "prompts": [
                "the same house wall is visibly divided into a large masonry section and a large wooden section",
                "the same wall plane has concrete masonry on the lower half and wooden boards on the upper half",
                "a house wall where brick masonry and woven bamboo each cover substantial wall areas",
                "a residential wall constructed half from plastered bricks and half from wooden planks",
                "one continuous house wall combining large concrete sections with large bamboo panels",
                "a house facade where both masonry and wood are dominant structural wall materials",
                "a wall with clearly connected brick and bamboo sections forming the same building wall",
                "a mixed material wall where neither masonry nor wood or bamboo is only a small incidental object",
            ],
        },

        "kayu_bambu": {
            "skor": 0,
            "nama": "Kayu biasa / bambu / rumbia",
            "prompts": [
                "a house wall dominated by woven bamboo panels",
                "a traditional Indonesian house with a full woven bamboo wall",
                "a simple rural dwelling whose main wall surface is woven bamboo",
                "a close-up photo of a gedek woven bamboo house wall",
                "a house facade mostly made of woven bamboo with small unrelated bricks nearby",
                "a bamboo house wall with incidental masonry objects in the foreground",
                "a house whose structural walls are predominantly bamboo despite a small concrete foundation",
                "a simple house wall entirely made of wooden boards",
                "a rural house facade dominated by horizontal wooden planks",
                "a wooden or bamboo house wall with a door window and small masonry base",
            ],
        },
    },

    "lantai": {
        "keramik": {
            "skor": 3,
            "nama": "Marmer / keramik",
            "prompts": [
                "an indoor house floor covered with ceramic tiles",
                "a residential floor made of square glossy ceramic tiles",
                "a clean tiled floor with clearly visible grout lines",
                "a house floor covered with patterned ceramic tiles",
                "a polished porcelain tile floor inside a house",
                "a polished marble slab floor inside a house",
                "a shiny smooth residential floor made of ceramic material",
                "a tiled indoor floor with repeated manufactured patterns",
                "a smooth finished floor covered by white ceramic tiles",
                "a decorative ceramic tile floor with straight regular joints",
            ],
        },

        "semen": {
            "skor": 1,
            "nama": "Semen / aci / bata",
            "prompts": [
                "a hard solid indoor floor made of unfinished gray cement",
                "a continuous concrete floor surface inside a house",
                "a smooth cement screed floor without ceramic tiles",
                "a rough but solid concrete slab floor",
                "an unpainted gray cement floor with cracks and stains",
                "a hard compact masonry floor made of exposed bricks",
                "an indoor floor paved with red clay bricks",
                "a brick floor arranged in a regular herringbone pattern",
                "a solid unfinished cement floor with a flat continuous surface",
                "a house floor made from hardened concrete rather than loose soil",
            ],
        },

        "tanah_kayu": {
            "skor": 0,
            "nama": "Tanah / kayu",
            "prompts": [
                "an indoor house floor made of bare natural earth",
                "a room with an unpaved dirt floor",
                "a house interior with loose dry soil covering the floor",
                "an uneven bare earth floor with footprints and loose dust",
                "an indoor floor covered with loose sand and natural soil",
                "a rough granular dirt floor without a solid concrete slab",
                "a rural room with a dusty unpaved ground surface",
                "a classroom with a bare dirt and sand floor",
                "an unfinished indoor ground floor made directly from soil",
                "a loose sandy floor with irregular footprints and no paving",
                "an indoor house floor made entirely of wooden planks",
                "a simple residential floor constructed from timber boards",
                "a traditional wooden plank floor inside a rural house",
                "an old house floor made of parallel wooden boards",
                "a raised timber floor with visible gaps between wooden planks",
                "a rough unfinished wood board floor inside a simple dwelling",
            ],
        },
    },

    "air": {
        "air_sangat_layak": {
            "skor": 3,
            "nama": "PDAM / air mineral kemasan",
            "prompts": [
                "a household PDAM municipal water meter and water pipe",
                "sealed branded bottled mineral drinking water",
                "a municipal PDAM water meter with an analog dial connected to household pipes",
                "a protected municipal water meter installed outside a residential house",
                "a close photo of an official piped water utility meter and valve",
                "a household municipal water connection with a visible meter gauge",
                "factory sealed mineral water bottles with intact caps and labels",
                "sealed packaged drinking water bottles stored inside a house",
                "factory sealed drinking water gallons with sealed caps",
                "multiple unopened packaged mineral water containers ready for drinking",
            ],
        },

        "air_layak": {
            "skor": 1,
            "nama": "Pompa / sumur terlindung / isi ulang",
            "prompts": [
                "an electric household water pump connected to pipes",
                "a protected household water well",
                "a refillable drinking water gallon container",
                "an electric household groundwater pump connected to PVC pipes",
                "a domestic electric water pump installed beside a house",
                "a protected dug well with a raised concrete ring and secure cover",
                "a clean protected household well with a concrete apron and drainage",
                "a reusable refill drinking water gallon mounted on a dispenser",
                "an unbranded refillable water gallon used inside a household",
                "a water refill container placed on a simple drinking water dispenser",
                "a protected groundwater source using a household electric pump",
                
            ],
        },

        "air_tidak_layak": {
            "skor": 0,
            "nama": "Sumber air tidak layak / lainnya",
            "prompts": [
                "an unprotected dirty water well",
                "people collecting household water from a river lake or swamp",
                "rainwater collected as the main household water source",
                "an unprotected open dirty water well exposed to soil and leaves",
                "a household collecting untreated water from a muddy river",
                "an unsafe shallow water hole with visibly contaminated stagnant water",
                "an open water source without a concrete cover or sanitary protection",
                "a bucket collecting raw water from a river lake swamp or pond",
                "an untreated muddy surface water source used by a household",
                "an unprotected spring surrounded by mud and contamination",
                "rainwater collected in an open dirty household container",
            ],
        },
    },

    "wc": {
        "wc_duduk": {
            "skor": 3,
            "nama": "WC duduk",
            "prompts": [
                "a western seated toilet bowl with a toilet seat",
                "a ceramic sitting toilet in a bathroom",
                "a ceramic sitting toilet with a flush tank in a bathroom",
                "a ceramic sitting toilet with closed lid in a bathroom",
                "a white ceramic seated toilet clearly visible inside a bathroom",
                "a modern sitting toilet with an oval seat lid and water tank",
                "a floor mounted western toilet standing upright on a raised pedestal",
                "a white flush toilet with a rectangular cistern tank behind the seat",
                "a chair height ceramic toilet designed for a person to sit on",
                "a raised toilet bowl with a closed oval lid and tall ceramic base",
                "a pedestal toilet standing above the tiled bathroom floor",
                "a complete sitting toilet consisting of a bowl seat lid tank and pedestal",
                "a front view of a white western seated toilet in a tiled bathroom",
                "a compact seated toilet with an elevated bowl and visible water cistern",
                "an Indonesian bathroom containing a modern white sitting toilet",
                "a bathroom toilet with a raised ceramic body rather than a floor level pan",
                "a white tank toilet installed upright against the bathroom wall",
                "a ceramic sitting toilet surrounded by tiled walls and tiled flooring",
                "a modern bathroom with a clearly visible pedestal sitting toilet",
                "a western style toilet bowl with a seat lid flush tank and tall base",
            ],
        },

        "wc_jongkok": {
            "skor": 1,
            "nama": "WC jongkok",
            "prompts": [
                "an Asian squat toilet pan built into a bathroom floor",
                "an Indonesian ceramic squatting toilet",
                "a ceramic squat toilet pan installed flat into the bathroom floor",
                "a low floor level squatting toilet with two foot placement areas",
                "an Asian squat toilet embedded horizontally in tiled flooring",
                "a flat oval squat toilet viewed from above",
                "a bathroom floor containing a ceramic squatting pan",
                "an Indonesian squat toilet installed on a low tiled platform",
                "a horizontal toilet pan designed to be used while squatting",
                "a floor level squat toilet without an elevated ceramic body",
                ],
        },

        "wc_lainnya": {
            "skor": 0,
            "nama": "Selain WC duduk/jongkok",
            "prompts": [
                "a bathroom without a visible toilet bowl",
                "an outdoor place with no toilet facility",
                "an open sewer used as a sanitation facility",
                "an outdoor pit latrine made from wood",
                "a riverbank used for open defecation",
                "a bucket used as an emergency sanitation facility",
                "an outdoor field with no sanitation structure",
                "a simple floor drain used for washing",
                "a simple outdoor MCK sanitation hut built above a river",
                "a traditional riverside latrine built from bamboo and fabric",
                "a small outdoor toilet enclosure standing over water",
                "a makeshift toilet hut built directly above a river",
                "a floating riverside sanitation facility without a visible toilet bowl",
                "a basic communal MCK structure beside a river",
                "an outdoor bamboo latrine used for defecation over water",
                "a temporary sanitation cubicle constructed above a pond",
            ],
        },
    },
}


KOMPONEN_FOTO = {
    "atap": [
        "a close-up photo of house roof material",
        "the underside of a roof showing roofing sheets or roof tiles",
        "a photograph focused on the roof of a house",
        "the upper roof covering of a residential building",
        "a sloped house roof viewed from outside",
        "a traditional thatched roof made from dried palm leaves",
        "a rumbia leaf roof covering the top of a rural house",
        "a roof made of layered dried leaves supported by wooden rafters",
        "the exterior top covering of a house, not a wall",
    ],

    "dinding": [
        "a close-up photo of a house wall surface",
        "an interior or exterior wall of a house",
        "the exterior facade of a house dominated by its wall",
        "a house wall made of woven bamboo panels",
        "a traditional Indonesian house with woven bamboo walls",
        "a house wall made of horizontal wooden planks",
        "a simple rural house with wooden or bamboo walls",
        "a masonry brick concrete or plastered house wall",
        "a vertical exterior wall of a house",
        "a vertical house wall with a door or window",
        "a masonry, wooden, or woven bamboo wall standing upright",
        "the side wall of a building, not the sloped roof",
    ],

    "lantai": [
        "a close-up photo of an indoor floor surface",
        "house flooring viewed from above",
        "a room where the floor surface is clearly visible",
    ],

    "air": [
        "a close-up photo of an electric household water pump connected to pipes",
        "a close-up photo of a municipal PDAM water meter",
        "a protected household water well",
        "a refillable drinking water gallon container",
        "sealed bottled mineral drinking water",
        "a river lake or spring used as a household water source",
    ],

    "wc": [
        "a photo of a toilet bowl or squat toilet inside a bathroom",
        "a bathroom photo centered on the toilet fixture",
        "a raised sitting toilet inside a bathroom",
        "a floor level squat toilet inside a bathroom",
        "a photo of a toilet bowl or squat toilet inside a bathroom",
        "a bathroom photo centered on the toilet fixture",
        "a raised sitting toilet inside a bathroom",
        "a floor level squat toilet inside a bathroom",
        "a simple outdoor MCK sanitation hut built above a river",
        "a traditional riverside latrine made from bamboo",
        "an outdoor toilet enclosure standing directly over water",
        "a makeshift communal sanitation facility beside a river",
    ],

    "lainnya": [
        "a photo unrelated to roof wall floor water source or toilet",
        "a person furniture vehicle food animal or random object",
    ],
}


_model = None
_preprocess = None
_tokenizer = None

_text_features = {}
_component_features = None


def normalisasi_fitur(features):
    """
    Menormalisasi tensor fitur tanpa operasi in-place.
    """
    norma = features.norm(
        dim=-1,
        keepdim=True,
    )

    norma = norma.clamp_min(1e-12)

    return features / norma


def muat_model():
    """
    Memuat OpenCLIP satu kali selama aplikasi hidup.
    """
    global _model
    global _preprocess
    global _tokenizer

    if _model is not None:
        return

    _model, _, _preprocess = (
        open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
            device=DEVICE,
        )
    )

    _tokenizer = open_clip.get_tokenizer(
        MODEL_NAME
    )

    _model.eval()


def fitur_teks(task):
    """
    Membuat satu prototipe teks untuk setiap kategori.
    Ini mempertahankan perilaku versi stabil awal.
    """
    if task in _text_features:
        return _text_features[task]

    labels = list(KATEGORI[task])
    semua_fitur = []

    with torch.no_grad():
        for label in labels:
            prompts = KATEGORI[task][label][
                "prompts"
            ]

            tokens = _tokenizer(
                prompts
            ).to(DEVICE)

            features = _model.encode_text(
                tokens
            )

            features = normalisasi_fitur(
                features
            )

            features = features.mean(
                dim=0,
                keepdim=True,
            )

            features = normalisasi_fitur(
                features
            )[0]

            semua_fitur.append(
                features.detach().clone()
            )

    text_features = torch.stack(
        semua_fitur
    )

    hasil = (labels, text_features)
    _text_features[task] = hasil

    return hasil


def fitur_komponen():
    """
    Membuat fitur teks untuk validasi jenis foto.
    """
    global _component_features

    if _component_features is not None:
        return _component_features

    labels = list(KOMPONEN_FOTO)
    semua_fitur = []

    with torch.no_grad():
        for label in labels:
            tokens = _tokenizer(
                KOMPONEN_FOTO[label]
            ).to(DEVICE)

            features = _model.encode_text(
                tokens
            )

            features = normalisasi_fitur(
                features
            )

            features = features.mean(
                dim=0,
                keepdim=True,
            )

            features = normalisasi_fitur(
                features
            )[0]

            semua_fitur.append(
                features.detach().clone()
            )

    fitur_tersusun = torch.stack(
        semua_fitur
    )

    _component_features = (
        labels,
        fitur_tersusun,
    )

    return _component_features


def validasi_komponen(
    task,
    image_features,
):
    """
    Memastikan foto sesuai dengan kolom yang dipilih.
    """
    labels, features = fitur_komponen()

    logits = (
        100.0
        * image_features
        @ features.T
    )

    probabilities = logits.softmax(
        dim=-1
    )[0]

    known_labels = [
        label
        for label in labels
        if label != "lainnya"
    ]

    known_indexes = [
        labels.index(label)
        for label in known_labels
    ]

    known_logits = logits[
        :,
        known_indexes,
    ]

    known_probabilities = (
        known_logits.softmax(dim=-1)[0]
    )

    indeks_known = int(
        known_probabilities.argmax().item()
    )

    terdeteksi = known_labels[
        indeks_known
    ]

    confidence = float(
        known_probabilities[
            indeks_known
        ].item()
    )

    expected_index = known_labels.index(
        task
    )

    expected_confidence = float(
        known_probabilities[
            expected_index
        ].item()
    )

    indeks_lainnya = labels.index(
        "lainnya"
    )

    other_confidence = float(
        probabilities[
            indeks_lainnya
        ].item()
    )

    if other_confidence >= 0.85:
        status = "ditolak"
        komponen_terdeteksi = "lainnya"

    elif terdeteksi == task:
        status = "diterima"
        komponen_terdeteksi = terdeteksi

    elif expected_confidence >= 0.20:
        status = "perlu_verifikasi"
        komponen_terdeteksi = terdeteksi

    else:
        status = "ditolak"
        komponen_terdeteksi = terdeteksi

    return {
        "diterima": status != "ditolak",
        "status": status,
        "komponen_diminta": task,
        "komponen_terdeteksi": (
            komponen_terdeteksi
        ),
        "keyakinan_terdeteksi": round(
            confidence,
            4,
        ),
        "keyakinan_komponen_diminta": round(
            expected_confidence,
            4,
        ),
        "keyakinan_lainnya": round(
            other_confidence,
            4,
        ),
        "semua_probabilitas": {
            labels[i]: round(
                float(
                    probabilities[i].item()
                ),
                4,
            )
            for i in range(len(labels))
        },
    }


def klasifikasi(
    task,
    image_path,
):
    """
    Mengklasifikasikan satu foto kondisi rumah.
    """
    if task not in KATEGORI:
        raise ValueError(
            f"Task tidak dikenal: {task}"
        )

    path = resolve_image_path(image_path)

    awal_total = time.perf_counter()

    muat_model()

    labels, text_features = fitur_teks(
        task
    )

    try:
        with Image.open(path) as gambar:
            gambar_rgb = gambar.convert("RGB")

            image = _preprocess(
                gambar_rgb
            ).unsqueeze(0).to(DEVICE)

    except Exception as error:
        raise ValueError(
            f"Foto tidak dapat dibaca: {path}"
        ) from error

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    awal_inferensi = time.perf_counter()

    with torch.no_grad():
        image_features = _model.encode_image(
            image
        )

        image_features = normalisasi_fitur(
            image_features
        )

        validasi = validasi_komponen(
            task,
            image_features,
        )

        probabilities = (
            TEMPERATURE_KATEGORI
            * image_features
            @ text_features.T
        ).softmax(dim=-1)[0]

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    durasi_inferensi = (
        time.perf_counter()
        - awal_inferensi
    )

    durasi_total = (
        time.perf_counter()
        - awal_total
    )

    if validasi["status"] == "ditolak":
        return {
            "task": task,
            "model": (
                f"OpenCLIP {MODEL_NAME} "
                f"({PRETRAINED})"
            ),
            "device": DEVICE,
            "status": "foto_tidak_sesuai",
            "hasil": None,
            "nama": None,
            "skor": None,
            "validasi_foto": validasi,
            "keyakinan_relatif": None,
            "selisih_probabilitas": None,
            "semua_probabilitas": None,
            "durasi_inferensi_detik": round(
                durasi_inferensi,
                4,
            ),
            "durasi_total_detik": round(
                durasi_total,
                4,
            ),
        }

    indeks = int(
        probabilities.argmax().item()
    )

    label = labels[indeks]
    data = KATEGORI[task][label]

    semua_probabilitas = {
        labels[i]: round(
            float(
                probabilities[i].item()
            ),
            4,
        )
        for i in range(len(labels))
    }

    return {
        "task": task,
        "model": (
            f"OpenCLIP {MODEL_NAME} "
            f"({PRETRAINED})"
        ),
        "device": DEVICE,
        "status": (
            "perlu_verifikasi"
            if validasi["status"]
            == "perlu_verifikasi"
            else "berhasil"
        ),
        "hasil": label,
        "nama": data["nama"],
        "skor": data["skor"],
        "validasi_foto": validasi,
        "keyakinan_relatif": round(
            float(
                probabilities[indeks].item()
            ),
            4,
        ),
        "semua_probabilitas": (
            semua_probabilitas
        ),
        "durasi_inferensi_detik": round(
            durasi_inferensi,
            4,
        ),
        "durasi_total_detik": round(
            durasi_total,
            4,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Klasifikasi cepat kondisi rumah "
            "dengan OpenCLIP"
        )
    )

    parser.add_argument(
        "--task",
        required=True,
        choices=list(KATEGORI),
    )

    parser.add_argument(
        "--image",
        required=True,
    )

    args = parser.parse_args()

    hasil = klasifikasi(
        task=args.task,
        image_path=args.image,
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