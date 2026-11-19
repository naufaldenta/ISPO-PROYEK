"""Klip bukti sebagai MP4 untuk laporan (wali kelas, kepala sekolah): kerangka dan pita keterangan ikut tergambar.

Video 10 fps; untuk tiap selang 0,1 dtk dipakai frame klip terbaru (frame tidak diinterpolasi). Codec "mp4v" dari
OpenCV, bisa diputar Windows Media Player dan VLC.
"""
from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from . import zone as zones
from .engine import Overlay, _draw

FPS = 10.0
BAND_H = 26


def _ascii(text: str) -> str:
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in text)


def _annotate(frame: np.ndarray, people: list | None, zone) -> np.ndarray:
    if people is None:
        return frame                                   # klip versi 1: kerangka sudah tergambar
    canvas = frame.copy()
    zones.draw(canvas, zone)
    for p in people:
        ov = Overlay(int(p["id"]), np.asarray(p["kp"], dtype=np.float32), np.asarray(p["conf"], dtype=np.float32),
                     p.get("warna", "green"), p.get("label", ""))
        _draw(canvas, ov)
    return canvas


def _decode(jpeg: bytes):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def render(frames: list[tuple[float, bytes]], overlay: list | None, zone, caption: str) -> bytes:
    """MP4 dari frame klip (detik, JPEG) + overlay versi 2 (atau None) + pita keterangan di bawah gambar.

    Frame dibuka satu per satu (hanya frame yang sedang ditulis yang ada di memori): klip 1080p bisa ratusan MB bila
    seluruhnya dibuka sekaligus.
    """
    first = _decode(frames[0][1]) if frames else None
    if first is None:
        raise ValueError("Klip kosong.")
    h, w = first.shape[:2]
    text = _ascii(caption)
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    writer = None
    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h + BAND_H))
        if not writer.isOpened():
            raise OSError("Penulis MP4 tidak tersedia.")
        times = [t for t, _ in frames]
        n_ticks = int(times[-1] * FPS) + 1
        index, current, current_index = 0, first, 0
        for k in range(n_ticks):
            tick = k / FPS
            while index + 1 < len(times) and times[index + 1] <= tick + 1e-6:
                index += 1
            if index != current_index:
                current, current_index = _decode(frames[index][1]), index
            if current is None or current.shape[:2] != (h, w):
                continue
            people = overlay[index]["orang"] if overlay and overlay[index] else None
            canvas = np.zeros((h + BAND_H, w, 3), np.uint8)
            canvas[:h] = _annotate(current, people if overlay else None, zone)
            cv2.putText(canvas, text, (6, h + BAND_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1,
                        cv2.LINE_AA)
            writer.write(canvas)
        writer.release()
        writer = None
        with open(path, "rb") as f:
            return f.read()
    finally:
        if writer is not None:
            writer.release()                     # Windows tidak bisa menghapus berkas yang masih terbuka
        try:
            os.unlink(path)
        except OSError:
            pass
