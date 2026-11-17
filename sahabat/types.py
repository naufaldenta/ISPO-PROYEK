"""Tipe data bersama. Titik tubuh memakai urutan COCO (17 titik) seperti keluaran YOLO-pose."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ViewMode = Literal["full", "upper", "none"]

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

SHOULDERS = (L_SH, R_SH)
ELBOWS = (L_EL, R_EL)
WRISTS = (L_WR, R_WR)
HIPS = (L_HIP, R_HIP)
KNEES = (L_KNEE, R_KNEE)
ANKLES = (L_ANK, R_ANK)

# Garis kerangka untuk digambar di layar.
SKELETON_EDGES = (
    (L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR),
    (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANK), (R_HIP, R_KNEE), (R_KNEE, R_ANK),
)


@dataclass(frozen=True, eq=False)
class Person:
    """Satu orang di satu frame."""
    track_id: int
    kp: np.ndarray       # (17, 2) float32, piksel
    conf: np.ndarray     # (17,) float32
    box: tuple[float, float, float, float]   # x1, y1, x2, y2


@dataclass(frozen=True, eq=False)
class Sample:
    """Satu pengamatan bercap waktu milik satu orang."""
    t: float
    kp: np.ndarray
    conf: np.ndarray
    size: float          # ukuran badan saat itu, piksel
    # Dihitung sekali saat sampel dibuat (lihat history.make_sample), dipakai berulang oleh penilai:
    center: np.ndarray | None = None      # tengah badan
    top: np.ndarray | None = None         # ujung atas ruas badan (hidung, atau tengah bahu)
    bottom: np.ndarray | None = None      # ujung bawah ruas badan (tengah pinggul)
