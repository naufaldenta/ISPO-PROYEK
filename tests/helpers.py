"""Pembuat kerangka sintetis untuk uji. Geometrinya dipakai semua uji, jangan diubah sembarangan."""
from __future__ import annotations

import dataclasses

import numpy as np

from sahabat.types import (L_ANK, L_EL, L_HIP, L_KNEE, L_SH, L_WR, NOSE, R_ANK, R_EL,
                           R_HIP, R_KNEE, R_SH, R_WR, Person)

LEGS = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK]


def skeleton(cx, hip_y, size=100.0, torso_deg=0.0, visible="full", r_wrist=None, r_ankle=None):
    """torso_deg: 0 tegak, 90 rebah ke kanan. visible: full | upper | none."""
    a = np.deg2rad(torso_deg)
    up = np.array([np.sin(a), -np.cos(a)])            # dari pinggul ke bahu
    side = np.array([np.cos(a), np.sin(a)])
    hip = np.array([cx, hip_y], dtype=float)
    sh = hip + up * size
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[NOSE] = sh + up * 0.3 * size
    for i, (du, ds) in zip((1, 2, 3, 4), ((0.05, -0.05), (0.05, 0.05), (0.0, -0.1), (0.0, 0.1))):
        kp[i] = kp[NOSE] + (up * du + side * ds) * size      # mata dan telinga di sekitar hidung
    kp[L_SH], kp[R_SH] = sh - side * 0.35 * size, sh + side * 0.35 * size
    kp[L_EL], kp[R_EL] = kp[L_SH] - up * 0.4 * size, kp[R_SH] - up * 0.4 * size
    kp[L_WR], kp[R_WR] = kp[L_SH] - up * 0.8 * size, kp[R_SH] - up * 0.8 * size
    kp[L_HIP], kp[R_HIP] = hip - side * 0.2 * size, hip + side * 0.2 * size
    kp[L_KNEE], kp[R_KNEE] = kp[L_HIP] - up * 0.9 * size, kp[R_HIP] - up * 0.9 * size
    kp[L_ANK], kp[R_ANK] = kp[L_HIP] - up * 1.8 * size, kp[R_HIP] - up * 1.8 * size
    if r_wrist is not None:
        kp[R_WR] = r_wrist
    if r_ankle is not None:
        kp[R_ANK] = r_ankle
    conf = np.ones(17, dtype=np.float32)
    if visible == "upper":
        conf[LEGS] = 0.0
    if visible == "none":
        conf[:] = 0.0
    return kp, conf


def person(track_id, **kw):
    kp, conf = skeleton(**kw)
    x1, y1 = kp.min(axis=0)
    x2, y2 = kp.max(axis=0)
    return Person(track_id, kp, conf, (float(x1), float(y1), float(x2), float(y2)))


def with_hidden_legs(p: Person) -> Person:
    conf = p.conf.copy()
    conf[LEGS] = 0.0
    return dataclasses.replace(p, conf=conf)


def ticks(t0=0.0, seconds=1.0, hz=10.0):
    """Cap waktu berjarak 1/hz, sebanyak seconds*hz."""
    return [t0 + i / hz for i in range(int(round(seconds * hz)))]


def feed(history, people_at, t0=0.0, seconds=1.0, hz=10.0):
    """people_at(t) -> list[Person]; memanggil history.update tiap 1/hz. Mengembalikan t terakhir."""
    t = t0
    for t in ticks(t0, seconds, hz):
        history.update(t, people_at(t))
    return t


def wait_for(pred, timeout_s: float) -> bool:
    """True begitu pred() benar, False bila waktu habis (dicek tiap 0,05 dtk)."""
    import time
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return bool(pred())
