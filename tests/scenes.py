"""Adegan dua orang sintetis, dipakai uji fitur pasangan, penilai kekerasan, mesin, dan pelatihan."""
from __future__ import annotations

import numpy as np

from dataclasses import replace

from sahabat.types import NOSE, R_ANK, R_EL, R_HIP, R_KNEE, R_SH, R_WR, Person

from .helpers import person

A_SHOULDER = np.array([285.0, 200.0])      # bahu kanan A (cx=250, size=100)
A_HANG = np.array([285.0, 280.0])          # pergelangan tangan kanan A saat menggantung
A_ANKLE = np.array([270.0, 480.0])         # pergelangan kaki kanan A saat berdiri
NEAR_B = np.array([372.5, 238.0])          # 30 piksel dari tengah badan B (400, 250)


def two(t, a_kw=None, b_kw=None):
    """A (nomor 1) di cx=250 dan B (nomor 2) di cx=400: jarak 1,5 ukuran badan."""
    return [person(1, **{**dict(cx=250, hip_y=300, size=100), **(a_kw or {})}),
            person(2, **{**dict(cx=400, hip_y=300, size=100), **(b_kw or {})})]


def limb_at(p, end, leg=False, bend=0):
    base, joint, tip = (R_HIP, R_KNEE, R_ANK) if leg else (R_SH, R_EL, R_WR)
    kp = p.kp.copy()
    kp[tip] = end
    kp[joint] = (kp[base] + kp[tip]) / 2 + np.array([0, bend])
    return replace(p, kp=kp)


def air_strike(t, leg=False, third=False):
    a, b = two(t)
    # Dari posisi menekuk, lurus mendekati B tanpa menyentuh (jarak 45 piksel).
    start = np.array([275, 351] if leg else [277, 220])
    end = np.array([355, 238])
    k = _ramp(t, 1.0, 0.2)
    a = limb_at(a, start + k * (end - start), leg=leg, bend=(1 - k) * 28)
    return [a, b] + ([person(3, cx=570, hip_y=300)] if third else [])


def hold_scene(t, zone="leher", seconds=2.5, hide_nose=False):
    a, b = two(t)
    if hide_nose:
        conf = b.conf.copy()
        conf[NOSE] = 0
        b = replace(b, conf=conf)
    if 1.0 <= t < 1.0 + seconds:
        end = {"leher": (400, 185), "badan": (400, 241), "bahu": (365, 200)}[zone]
        a = limb_at(a, end)
    return [a, b]


def _ramp(t, t0, dur):
    return min(max((t - t0) / dur, 0.0), 1.0)


def punch(t, t0=1.0):
    """Pergelangan tangan kanan A melesat dari bahunya ke dekat badan B dalam 0,3 dtk."""
    if t < t0:
        return two(t)
    return two(t, a_kw=dict(r_wrist=A_SHOULDER + (NEAR_B - A_SHOULDER) * _ramp(t, t0, 0.3)))


def kick(t, t0=1.0):
    if t < t0:
        return two(t)
    return two(t, a_kw=dict(r_ankle=A_ANKLE + (NEAR_B - A_ANKLE) * _ramp(t, t0, 0.3)))


def handshake(t):
    """Tangan A mendekat pelan (0,5 ukuran badan per detik)."""
    x = min(285.0 + 50.0 * t, 355.0)
    return two(t, a_kw=dict(r_wrist=np.array([x, 280.0])))


def push(t, t0=1.0):
    """Tangan A sudah menempel dekat B, lalu B terdorong menjauh 2 ukuran badan per detik."""
    return two(t, a_kw=dict(r_wrist=NEAR_B), b_kw=dict(cx=400 + 200 * min(max(t - t0, 0.0), 0.3)))


def repeated_punch(t):
    return punch(t, t0=1.0 + 0.6 * int(max(t - 1.0, 0.0) / 0.6))


def punch_then_still(t):
    return punch(t) if t < 1.6 else two(t)


def single_person(t):
    return [person(1, cx=250, hip_y=300)]


def mirrored(script):
    """Menukar peran: nomor 2 melakukan apa yang tadinya dilakukan nomor 1."""
    def flipped(t):
        out = []
        for p in script(t):
            kp = p.kp.copy()
            kp[:, 0] = 650.0 - kp[:, 0]
            x1, y1, x2, y2 = p.box
            out.append(Person({1: 2, 2: 1}.get(p.track_id, p.track_id), kp, p.conf,
                              (650.0 - x2, y1, 650.0 - x1, y2)))
        return sorted(out, key=lambda q: q.track_id)
    return flipped
