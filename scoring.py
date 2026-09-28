from typing import Optional


SKOR_ATAP = {
    "genteng_bagus": 3,
    "genteng_beton": 3,
    "beton": 3,
    "multiroof": 3,

    "genteng_biasa": 1,
    "genteng_tanah_liat": 1,
    "seng": 1,
    "asbes": 1,

    "rumbia": 0,
    "anyaman_daun": 0,
    "anyaman_dedaunan": 0,
}


SKOR_DINDING = {
    "tembok": 3,
    "bata": 3,
    "batako": 3,
    "kayu_jati_bagus": 3,

    "setengah_tembok": 1,
    "campuran_tembok_papan": 1,
    "campuran_tembok_bambu": 1,

    "kayu_biasa": 0,
    "bambu": 0,
    "rumbia": 0,
}


SKOR_LANTAI = {
    "marmer": 3,
    "keramik": 3,

    "semen_halus": 1,
    "semen_kasar": 1,
    "aci": 1,
    "bata": 1,

    "tanah": 0,
    "kayu": 0,
}


SKOR_AIR = {
    "pdam": 3,
    "air_kemasan": 3,

    "pompa_air": 1,
    "sumur_terlindung": 1,
    "air_isi_ulang": 1,
    "mata_air_terlindung": 1,

    "sumur_tidak_terlindung": 0,
    "air_hujan": 0,
    "sungai": 0,
    "danau": 0,
    "rawa": 0,
    "mata_air_tidak_terlindung": 0,
    "lainnya": 0,
}


SKOR_WC = {
    "wc_duduk": 3,
    "wc_jongkok": 1,
    "selainnya": 0,
}


def skor_luas(
    perkiraan_luas_m2: float,
    jumlah_anggota: int,
) -> dict:
    if jumlah_anggota <= 0:
        raise ValueError(
            "Jumlah anggota keluarga minimal 1."
        )

    # Ketentuan mentor:
    # AI hanya boleh menghasilkan maksimum 80 m².
    luas_dibatasi = max(
        1.0,
        min(float(perkiraan_luas_m2), 80.0),
    )

    luas_per_orang = (
        luas_dibatasi / jumlah_anggota
    )

    if luas_per_orang >= 14:
        skor = 3
    elif luas_per_orang >= 7:
        skor = 1
    else:
        skor = 0

    return {
        "perkiraan_luas_m2": round(
            luas_dibatasi,
            2,
        ),
        "jumlah_anggota": jumlah_anggota,
        "luas_per_orang_m2": round(
            luas_per_orang,
            2,
        ),
        "skor": skor,
    }


def ambil_skor(
    pemetaan: dict,
    hasil_ai: Optional[str],
) -> Optional[int]:
    if hasil_ai is None:
        return None

    hasil_ai = (
        str(hasil_ai)
        .strip()
        .lower()
        .replace(" ", "_")
    )

    if hasil_ai == "tidak_dapat_dinilai":
        return None

    return pemetaan.get(hasil_ai)


def skor_atap(hasil_ai: str) -> Optional[int]:
    return ambil_skor(
        SKOR_ATAP,
        hasil_ai,
    )


def skor_dinding(
    hasil_ai: str,
) -> Optional[int]:
    return ambil_skor(
        SKOR_DINDING,
        hasil_ai,
    )


def skor_lantai(
    hasil_ai: str,
) -> Optional[int]:
    return ambil_skor(
        SKOR_LANTAI,
        hasil_ai,
    )


def skor_air(hasil_ai: str) -> Optional[int]:
    return ambil_skor(
        SKOR_AIR,
        hasil_ai,
    )


def skor_wc(hasil_ai: str) -> Optional[int]:
    return ambil_skor(
        SKOR_WC,
        hasil_ai,
    )


def hitung_total(
    luas: Optional[int],
    atap: Optional[int],
    dinding: Optional[int],
    lantai: Optional[int],
    air: Optional[int],
    wc: Optional[int],
) -> dict:
    komponen = {
        "luas": luas,
        "atap": atap,
        "dinding": dinding,
        "lantai": lantai,
        "sumber_air": air,
        "wc": wc,
    }

    belum_lengkap = [
        nama
        for nama, nilai in komponen.items()
        if nilai is None
    ]

    subtotal = sum(
        nilai
        for nilai in komponen.values()
        if nilai is not None
    )

    if belum_lengkap:
        return {
            "skor": komponen,
            "subtotal": subtotal,
            "total": None,
            "maksimum": 18,
            "status": "belum_lengkap",
            "komponen_belum_lengkap": (
                belum_lengkap
            ),
        }

    return {
        "skor": komponen,
        "subtotal": subtotal,
        "total": subtotal,
        "maksimum": 18,
        "status": "lengkap",
        "komponen_belum_lengkap": [],
    }