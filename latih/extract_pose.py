"""Mengubah video dataset menjadi titik tubuh 10 FPS per klip, dengan pelacak yang sama seperti siaran langsung.

    python -m latih.extract_pose <folder_video> <folder_keluar> [resolusi]

Susunan subfolder dipertahankan, karena label diambil dari nama folder.
Boleh dihentikan dan dijalankan ulang: klip yang sudah ada dilewati.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from sahabat.types import Person

RATE_HZ = 10.0
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def save_clip(path: Path, frames: list[tuple[float, list[Person]]], frame_hw: tuple[int, int]) -> None:
    """frames: [(cap waktu, [Person, ...]), ...]. Frame tanpa orang tetap dicatat."""
    t = np.array([ft for ft, _ in frames], dtype=np.float64)
    frame_of, ids, kp, conf, box = [], [], [], [], []
    for i, (_, people) in enumerate(frames):
        for p in people:
            frame_of.append(i)
            ids.append(p.track_id)
            kp.append(p.kp)
            conf.append(p.conf)
            box.append(p.box)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, t=t, frame_hw=np.array(frame_hw, dtype=np.int32),
        frame_of=np.array(frame_of, dtype=np.int32), track_id=np.array(ids, dtype=np.int32),
        kp=np.array(kp, dtype=np.float32).reshape(-1, 17, 2),
        conf=np.array(conf, dtype=np.float32).reshape(-1, 17),
        box=np.array(box, dtype=np.float32).reshape(-1, 4))


def extract(video: Path, out: Path, pose) -> int:
    """Menyimpan titik tubuh satu video; mengembalikan jumlah frame yang dinilai."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"tidak bisa membuka {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1.0 <= fps <= 240.0 else 30.0
    pose.reset()                                   # nomor pelacak tidak boleh terbawa dari video sebelumnya
    frames, hw, index, last_slot = [], (0, 0), 0, -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index / fps
            index += 1
            slot = int(t * RATE_HZ + 1e-9)
            if slot == last_slot:                  # ambil satu frame per 0,1 detik
                continue
            last_slot = slot
            hw = frame.shape[:2]
            frames.append((round(slot / RATE_HZ, 3), pose.detect(frame)))
    finally:
        cap.release()
    save_clip(out, frames, hw)
    return len(frames)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    src, dst = Path(argv[0]), Path(argv[1])
    imgsz = int(argv[2]) if len(argv) > 2 else 480
    from sahabat.pose import YoloPose
    pose = YoloPose(imgsz)
    videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    print(f"{len(videos)} video, resolusi {imgsz}")
    started, done = time.monotonic(), 0
    for i, video in enumerate(videos, 1):
        out = (dst / video.relative_to(src)).with_suffix(".npz")
        if out.exists():
            continue
        try:
            n = extract(video, out, pose)
        except Exception as e:                     # satu video rusak tidak menghentikan semuanya
            print(f"  LEWATI {video.name}: {e}")
            continue
        done += 1
        if done % 10 == 0 or i == len(videos):
            print(f"  {i}/{len(videos)}  {video.name}: {n} frame  ({time.monotonic() - started:.0f} dtk)", flush=True)
    print("selesai")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
