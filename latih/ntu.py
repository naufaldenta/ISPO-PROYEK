"""Membaca kerangka 2D NTU RGB+D 120 (berkas ntu120_2d.pkl dari OpenMMLab) ke format klip kita.

Berkas itu berisi titik tubuh COCO-17 hasil HRNet dari video NTU, 30 FPS, paling banyak dua orang per
klip, lengkap dengan nomor aksi. Titik tubuhnya sama jenisnya dengan keluaran YOLO-pose, jadi klipnya
bisa langsung diputar lewat History dan penilai yang sama dengan aplikasi.

Berkas .pkl dari luar dibaca dengan pembatas: hanya larik numpy dan wadah bawaan Python yang boleh
dibentuk, sehingga isi berkas tidak bisa menjalankan kode.
"""
from __future__ import annotations

import importlib
import pickle
import re
from pathlib import Path

import numpy as np

from sahabat.types import Person

SRC_FPS = 30.0
RATE_HZ = 10.0
MIN_SCORE = 0.05           # di bawah ini titik dianggap tidak ada

# Aksi dua orang (nomor aksi NTU, mulai dari 1).
HARMFUL = {106: "memukul dengan benda", 107: "mengayunkan pisau", 108: "menjatuhkan orang",
           109: "merampas barang", 110: "menembak", 111: "menginjak kaki"}
VIOLENT = {50: "pukul/tampar", 51: "tendang", 52: "dorong", **HARMFUL}
FRIENDLY = {53: "tepuk punggung", 54: "menunjuk", 55: "pelukan", 56: "memberi barang", 57: "menyentuh saku",
            58: "salaman", 59: "berjalan mendekat", 60: "berjalan menjauh", 112: "tos", 113: "bersulang",
            114: "mengangkat barang bersama", 115: "memotret", 116: "mengikuti", 117: "berbisik",
            118: "bertukar barang", 119: "memapah", 120: "suit"}
# Aksi satu orang untuk menguji aturan jatuh.
FALL = {43: "jatuh"}
NOT_FALL = {8: "duduk", 9: "berdiri dari duduk", 6: "mengambil barang di lantai", 16: "memakai sepatu",
            26: "melompat satu kaki", 27: "melompat", 42: "terhuyung", 48: "mual/membungkuk", 80: "jongkok"}

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("collections", "OrderedDict"), ("collections", "defaultdict"),
    ("_codecs", "encode"),                      # cara lama numpy menyimpan bytes; hanya mengubah teks ke bytes
}
_NAME = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"isi berkas tidak diizinkan: {module}.{name}")
        return getattr(importlib.import_module(module.replace("numpy.core.", "numpy._core.")), name)


def safe_load(path: Path):
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


def parse_name(frame_dir: str) -> dict[str, int]:
    """'S001C002P003R002A050' -> setup, kamera, pemeran, ulangan, aksi."""
    m = _NAME.search(frame_dir)
    if not m:
        raise ValueError(f"nama klip tidak dikenal: {frame_dir}")
    return dict(zip(("setup", "camera", "performer", "replication", "action"), map(int, m.groups())))


def to_frames(keypoint: np.ndarray, score: np.ndarray, hold_last_s: float = 0.0) -> list[tuple[float, list[Person]]]:
    """keypoint (M, T, 17, 2), score (M, T, 17) pada 30 FPS -> [(t, [Person])] pada 10 FPS.

    hold_last_s: mengulang frame terakhir selama itu (klip jatuh NTU dipotong tepat setelah orangnya rebah).
    """
    step = int(round(SRC_FPS / RATE_HZ))
    people_n, total = keypoint.shape[0], keypoint.shape[1]
    frames = []
    for k, src in enumerate(range(0, total, step)):
        people = []
        for m in range(people_n):
            conf = score[m, src].astype(np.float32)
            seen = conf > MIN_SCORE
            if not seen.any():
                continue
            kp = keypoint[m, src].astype(np.float32)
            x1, y1 = kp[seen].min(axis=0)
            x2, y2 = kp[seen].max(axis=0)
            people.append(Person(m + 1, kp, conf, (float(x1), float(y1), float(x2), float(y2))))
        frames.append((round(k / RATE_HZ, 3), people))
    if frames and hold_last_s > 0:
        last_t, last_people = frames[-1]
        for j in range(1, int(round(hold_last_s * RATE_HZ)) + 1):
            frames.append((round(last_t + j / RATE_HZ, 3), last_people))
    return frames
