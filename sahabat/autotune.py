"""Mengukur kecepatan laptop dan memilih resolusi, supaya tidak ada yang perlu diatur manual."""
from __future__ import annotations

import dataclasses
import platform
import time
from typing import Callable

import numpy as np

from .config import Config

# Pengukuran hanya inferensi pose. Tiap putaran juga menghitung fitur dan memanggil dua model penilai
# (terukur ~19 ms untuk dua orang, ~38 ms untuk empat orang), plus kamera dan gambar.
SCORING_OVERHEAD_S = 0.030


def pipeline_fps(inference_fps: float) -> float:
    """Perkiraan laju seluruh putaran dari laju inferensi pose saja."""
    if inference_fps <= 0:
        return 0.0
    return 1.0 / (1.0 / inference_fps + SCORING_OVERHEAD_S)


def choose_imgsz(fps_by_size: dict[int, float], target: float, floor: float) -> tuple[int, bool]:
    """(resolusi, lambat?). Terbesar yang mencapai target; kalau tidak ada, terbesar yang mencapai lantai."""
    sizes = sorted(fps_by_size, reverse=True)
    for limit in (target, floor):
        for size in sizes:
            if fps_by_size[size] >= limit:
                return size, False
    return sizes[-1], True


def measure(make_pose: Callable[[int], object], sizes: tuple[int, ...], frame: np.ndarray,
            seconds: float = 3.0) -> dict[int, float]:
    """FPS inferensi per resolusi pada satu frame."""
    result = {}
    for size in sizes:
        pose = make_pose(size)
        for _ in range(3):                       # pemanasan
            pose.detect(frame)
        count, started = 0, time.perf_counter()
        while True:
            pose.detect(frame)
            count += 1
            elapsed = time.perf_counter() - started
            if elapsed >= seconds:
                break
        result[size] = count / elapsed
    return result


def ensure_tuned(cfg: Config, frame: np.ndarray, make_pose: Callable[[int], object] | None = None,
                 seconds: float = 3.0) -> Config:
    """Mengukur ulang hanya bila belum pernah, atau folder ini dipindah ke laptop lain."""
    node = platform.node()
    if cfg.imgsz is not None and cfg.tuned_for == node:
        return cfg
    if make_pose is None:
        from .pose import YoloPose
        make_pose = YoloPose
    fps = measure(make_pose, cfg.imgsz_candidates, frame, seconds)
    size, slow = choose_imgsz({s: pipeline_fps(v) for s, v in fps.items()}, cfg.fps_target, cfg.fps_floor)
    return dataclasses.replace(cfg, imgsz=size, slow=slow, tuned_for=node)
