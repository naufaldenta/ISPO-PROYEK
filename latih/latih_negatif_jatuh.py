"""Latih penilai kekerasan dengan contoh negatif dari adegan jatuh mandiri.

python -m latih.latih_negatif_jatuh DATA OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .latih_kekerasan_baru import Replay, core
from .latih_rlvs import items
from .putar import PoseTrace, load_config_snapshot
from .violence_train import Item, fit, stack, trace_pairs


def installed_sources(data: Path, out: Path, cfg) -> list[tuple]:
    """Baris latih model kekerasan terpasang (rlvs-hard1-fall): (X, y, bobot) per sumber."""
    labels = json.loads((out / "violence-fall-negative-labels.json").read_text(encoding="utf-8"))
    base = core(data, cfg)
    rlvs = [r.item for r in items(out, "train", cfg)]
    # Replay tetap diimpor untuk membaca cache AIRTLab lama yang memakai __main__.
    ordinary = [r.item for r in joblib.load(out / "airtlab-replay.joblib")
                if r.split == "train" and not r.item.violent]
    tr = PoseTrace.load(data / "webcam/sesi-20261115-211259/jejak_416.npz")
    falls = []
    for group, (start, end) in enumerate(labels["webcam_ranges"], 9000):
        seconds = tr.t - tr.t[0]
        indices = np.flatnonzero((seconds >= start - labels["history_preroll_s"]) & (seconds <= end))
        trace = PoseTrace(tr.t[indices], tr.frame_h, tr.frame_w, [tr.people[i] for i in indices])
        falls.append(Item("webcam", f"jatuh-mandiri-{group - 9000}", False, group, trace_pairs(trace, cfg)))
    gmdcsa = []
    for group, name in enumerate(labels["gmdcsa_clips"], 10000):
        trace = PoseTrace.load(data / "gmdcsa24/jejak_416" / f"{name}.npz")
        gmdcsa.append(Item("gmdcsa-negative", name, False, group, trace_pairs(trace, cfg)))
    sources = [base, stack(rlvs, 3), stack(ordinary, 1), stack(falls, labels["webcam_weight"])]
    if gmdcsa:
        sources.append(stack(gmdcsa, labels["gmdcsa_weight"]))
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    cfg = load_config_snapshot(args.out / "config-baseline.json")
    x, y, weights = [np.concatenate(parts) for parts in zip(*installed_sources(args.data, args.out, cfg), strict=True)]
    print(f"{len(y)} baris latih", flush=True)
    model = fit(x, y, weights)
    target = args.out / "violence-rlvs-hard1-fall.joblib"
    temporary = target.with_suffix(".part")
    joblib.dump(model, temporary, compress=3)
    temporary.replace(target)
    print(target, flush=True)


if __name__ == "__main__":
    main()
