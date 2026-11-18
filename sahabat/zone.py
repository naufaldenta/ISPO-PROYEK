"""Zona pantau: hanya orang yang titik tengah badannya di dalam zona yang dinilai.

Zona disimpan sebagai poligon dengan koordinat ternormalisasi (0..1 terhadap lebar dan tinggi gambar), supaya
tetap berlaku bila resolusi kamera berubah. None berarti seluruh gambar (bawaan, tanpa pengaturan apa pun).
"""
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .types import Person

MIN_POINTS = 3
MAX_POINTS = 24

Zone = tuple[tuple[float, float], ...]


def normalize(points) -> Zone | None:
    """Poligon yang sah, atau None (seluruh gambar). ValueError bila bentuknya salah."""
    if points is None:
        return None
    if not isinstance(points, (list, tuple)) or not MIN_POINTS <= len(points) <= MAX_POINTS:
        raise ValueError(f"Zona butuh {MIN_POINTS} sampai {MAX_POINTS} titik.")
    out = []
    for p in points:
        if (not isinstance(p, (list, tuple)) or len(p) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in p)):
            raise ValueError("Setiap titik zona harus berupa [x, y].")
        x, y = float(p[0]), float(p[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("Titik zona harus di dalam gambar.")
        out.append((round(x, 4), round(y, 4)))
    if abs(_area(out)) < 1e-3:
        raise ValueError("Zona terlalu kecil.")
    return tuple(out)


def load(points) -> Zone | None:
    """Zona dari berkas pengaturan; isi yang rusak berarti seluruh gambar."""
    try:
        return normalize(points)
    except ValueError:
        return None


def contains(zone: Zone | None, x: float, y: float) -> bool:
    """Titik ternormalisasi (x, y) di dalam zona. Tanpa zona: selalu benar."""
    if zone is None:
        return True
    inside = False
    n = len(zone)
    for i in range(n):
        x1, y1 = zone[i]
        x2, y2 = zone[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def center(p: Person) -> tuple[float, float]:
    x1, y1, x2, y2 = p.box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def split(people: Sequence[Person], zone: Zone | None, shape) -> tuple[list[Person], list[Person]]:
    """(di dalam zona, di luar zona). shape = (tinggi, lebar) gambar."""
    if zone is None:
        return list(people), []
    h, w = shape[:2]
    inside, outside = [], []
    for p in people:
        cx, cy = center(p)
        (inside if contains(zone, cx / w, cy / h) else outside).append(p)
    return inside, outside


def draw(canvas: np.ndarray, zone: Zone | None) -> None:
    """Menggelapkan bagian di luar zona dan menggambar garis tepinya, langsung di canvas."""
    if zone is None:
        return
    h, w = canvas.shape[:2]
    pts = np.array([[round(x * w), round(y * h)] for x, y in zone], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    outside = mask == 0
    canvas[outside] = (canvas[outside] * 0.55).astype(np.uint8)
    cv2.polylines(canvas, [pts], True, (59, 212, 255), 2, cv2.LINE_AA)


def _area(points) -> float:
    s = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        s += x1 * y2 - x2 * y1
    return s / 2.0
