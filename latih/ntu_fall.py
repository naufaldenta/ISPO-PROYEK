"""Menguji aturan jatuh pada klip satu-orang NTU: A043 (jatuh) melawan gerakan sehari-hari.

    python -m latih.ntu_fall <ntu120_2d.pkl> [jumlah_pekerja]

Klip jatuh NTU dipotong tepat setelah orangnya rebah, padahal aturan menunggu orang itu tetap rendah.
Karena itu frame terakhir SEMUA klip (jatuh maupun bukan) diulang 1,5 detik, seolah orangnya diam.

Pemeran bernomor ganjil dipakai untuk menyetel, yang genap untuk menguji.
"""
from __future__ import annotations

import dataclasses
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from sahabat.config import Config
from sahabat.fall import FallDetector
from sahabat.history import History

from .ntu import FALL, NOT_FALL, parse_name, safe_load, to_frames

HOLD_S = 1.5
# Varian ambang yang dibandingkan; yang pertama adalah nilai bawaan aplikasi.
VARIANTS: tuple[tuple[str, dict], ...] = (
    ("bawaan", {}),
    ("pinggul 0.7", {"fall_hip_speed": 0.7}),
    ("pinggul 0.5", {"fall_hip_speed": 0.5}),
    ("rebah 50", {"fall_lying_deg": 50.0}),
    ("rebah 50 + pinggul 0.7", {"fall_lying_deg": 50.0, "fall_hip_speed": 0.7}),
    ("rebah 45 + pinggul 0.5", {"fall_lying_deg": 45.0, "fall_hip_speed": 0.5}),
)


def fall_fired(frames, cfg: Config, frame_h: int) -> bool:
    """Apakah aturan jatuh memicu pada klip ini (riwayat dan detektor yang sama dengan aplikasi)."""
    history, detector = History(cfg), FallDetector(cfg)
    for t, people in frames:
        history.update(t, people)
        if detector.tick(t, history.tracks, frame_h):
            return True
    return False


def _work(item):
    keypoint, score, frame_h = item
    frames = to_frames(keypoint, score, hold_last_s=HOLD_S)
    return [fall_fired(frames, dataclasses.replace(Config(), **override), frame_h) for _, override in VARIANTS]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    workers = int(argv[1]) if len(argv) > 1 else 4
    started = time.monotonic()
    names = {**FALL, **NOT_FALL}
    annotations = [a for a in safe_load(Path(argv[0]))["annotations"] if a["label"] + 1 in names]
    meta = [parse_name(a["frame_dir"]) for a in annotations]
    items = [(a["keypoint"], a["keypoint_score"], a["img_shape"][0]) for a in annotations]
    del annotations
    with Pool(workers) as pool:
        results = pool.map(_work, items, chunksize=64)

    for split, wanted in (("PENYETELAN (pemeran ganjil)", 1), ("UJI (pemeran genap)", 0)):
        print(f"\n{split}: klip yang memicu alarm jatuh")
        header = "  aksi".ljust(34) + "".join(name.rjust(26) for name, _ in VARIANTS)
        print(header)
        for action, label in names.items():
            idx = [i for i, m in enumerate(meta) if m["action"] == action and m["performer"] % 2 == wanted]
            cells = []
            for v in range(len(VARIANTS)):
                hit = sum(results[i][v] for i in idx)
                cells.append(f"{hit}/{len(idx)} ({100 * hit / max(len(idx), 1):3.0f}%)".rjust(26))
            print(f"  A{action:03d} {label}".ljust(34) + "".join(cells))
    print(f"\n({time.monotonic() - started:.0f} dtk)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
