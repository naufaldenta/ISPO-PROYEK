"""Menghitung fitur pasangan untuk klip dua-orang NTU sekali saja, lalu menyimpannya.

    python -m latih.ntu_build <ntu120_2d.pkl> <keluaran.npz> [jumlah_pekerja] [--reuse cache.npz]

Tiap klip diputar lewat History dan fitur yang sama dengan aplikasi. Hasilnya dipakai train_rf.
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sahabat.config import Config
from sahabat.pair_features import WINDOW_FEATURE_NAMES

from .ntu import FRIENDLY, RATE_HZ, VIOLENT, parse_name, safe_load, to_frames
from .windows import ClipPairs, pairs_from_frames

N_FEAT = len(WINDOW_FEATURE_NAMES)


def build_clip(keypoint: np.ndarray, score: np.ndarray, cfg: Config) -> ClipPairs:
    return pairs_from_frames(to_frames(keypoint, score), cfg)


def save_cache(path: Path, clips: list[ClipPairs], meta: list[dict]) -> None:
    rows = [r for c in clips for r in c.rows]
    starts = np.cumsum([0] + [len(c.rows) for c in clips])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n_frames=np.array([len(c.t) for c in clips], dtype=np.int32), row_start=starts.astype(np.int64),
        frame=np.array([r[0] for r in rows], dtype=np.int32),
        id_a=np.array([r[1] for r in rows], dtype=np.int32), id_b=np.array([r[2] for r in rows], dtype=np.int32),
        ab=np.array([r[3] for r in rows], dtype=np.float32).reshape(-1, N_FEAT),
        ba=np.array([r[4] for r in rows], dtype=np.float32).reshape(-1, N_FEAT),
        **{k: np.array([m[k] for m in meta], dtype=np.int32) for k in ("action", "performer", "setup", "camera")})


def load_cache(path: Path) -> tuple[list[ClipPairs], list[dict]]:
    with np.load(path) as npz:                      # tiap larik dibaca sekali; npz membuka ulang berkas tiap diakses
        d = {k: npz[k] for k in npz.files}
    frame, id_a, id_b = d["frame"].tolist(), d["id_a"].tolist(), d["id_b"].tolist()
    ab, ba, starts = d["ab"], d["ba"], d["row_start"].tolist()
    keys = ("action", "performer", "setup", "camera")
    cols = {k: d[k].tolist() for k in keys}
    clips, meta = [], []
    for i, n in enumerate(d["n_frames"].tolist()):
        rows = [(frame[j], id_a[j], id_b[j], ab[j], ba[j]) for j in range(starts[i], starts[i + 1])]
        clips.append(ClipPairs([round(k / RATE_HZ, 3) for k in range(n)], rows))
        meta.append({k: cols[k][i] for k in keys})
    return clips, meta


def _work(item):
    keypoint, score = item
    return build_clip(keypoint, score, Config())


def main(argv: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("workers", type=int, nargs="?", default=4)
    parser.add_argument("--reuse", type=Path, help="Cache sebelumnya dengan konfigurasi fitur yang sama")
    args = parser.parse_args(argv)
    wanted = set(VIOLENT) | set(FRIENDLY)
    clips, meta = load_cache(args.reuse) if args.reuse else ([], [])
    if any(m["action"] not in wanted for m in meta):
        parser.error("--reuse harus berisi aksi dalam daftar pelatihan ini")
    wanted -= {m["action"] for m in meta}
    started = time.monotonic()
    annotations = [a for a in safe_load(args.source)["annotations"] if a["label"] + 1 in wanted]
    for a in annotations:
        m = parse_name(a["frame_dir"])
        meta.append({k: m[k] for k in ("action", "performer", "setup", "camera")})
    print(f"{len(annotations)} klip dua-orang dipilih ({time.monotonic() - started:.0f} dtk untuk memuat)", flush=True)

    items = [(a["keypoint"], a["keypoint_score"]) for a in annotations]
    del annotations
    with Pool(args.workers) as pool:
        for i, clip in enumerate(pool.imap(_work, items, chunksize=32), 1):
            clips.append(clip)
            if i % 2000 == 0:
                print(f"  {i}/{len(items)}  ({time.monotonic() - started:.0f} dtk)", flush=True)
    save_cache(args.output, clips, meta)
    print(f"selesai: {len(clips)} klip, {sum(len(c.rows) for c in clips)} baris -> {args.output}"
          f"  ({time.monotonic() - started:.0f} dtk)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
