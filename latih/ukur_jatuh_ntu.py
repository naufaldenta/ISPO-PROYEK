"""Ukur perubahan deteksi jatuh pada jejak dataset yang tersedia di disk."""
from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sahabat.config import MODELS_DIR, Config
from sahabat.fall import FallDetector
from sahabat.history import EPS, History
from sahabat.person_features import person_features
from .ntu import FALL, NOT_FALL, safe_load, to_frames
from .putar import PoseTrace
from .urfall import hold

PUSH_UP = {"s2-adl-22", "s3-adl-06", "s4-adl-05", "s4-adl-06", "s4-adl-07"}


def compare(frames, height, model, cfg: Config):
    history = History(cfg)
    baseline = dataclasses.replace(cfg, fall_need_upright=False, fall_edge_frac=2.0)
    dets = [FallDetector(baseline), FallDetector(cfg)]
    # Uji offline: hitung peluang satu kali untuk seluruh klip, kemudian putar
    # putaran dengan urutan yang sama. Kedua penilai mendapat peluang identik.
    frames = list(frames)
    features, sizes = [], []
    for t, people in frames:
        history.update(float(t), people)
        size = len(features)
        for track in history.tracks.values():
            if track.mode == "full" and track.samples and t - track.samples[-1].t <= EPS:
                row = person_features(track, cfg.fall_window_s)
                if row is not None:
                    features.append(row)
        sizes.append(len(features) - size)
    probabilities = model.predict_proba(np.array(features)) if features else np.empty((0, 2))
    history = History(cfg)
    predictions = _Predictions()
    for det in dets:
        det._model = predictions
    events = [[], []]
    start = 0
    for (t, people), size in zip(frames, sizes):
        history.update(float(t), people)
        predictions.current = probabilities[start:start + size]
        start += size
        for out, det in zip(events, dets):
            out.extend((round(float(t), 3), i, det.sources[i])
                       for i in det.tick(float(t), history.tracks, height))
    return events


class _Predictions:
    def predict_proba(self, X):
        if len(X) != len(self.current):
            raise ValueError("Jumlah orang yang dinilai berubah pada replay")
        return self.current


def _init_ntu():
    global _model
    _model = FallDetector(Config(), MODELS_DIR / "fall_model.joblib")._model


def _ntu_job(clip):
    kp, conf, height, name, label = clip
    frames = to_frames(kp, conf, hold_last_s=1.5)
    before, after = compare(frames, height, _model, Config())
    return {"dataset": "ntu", "name": name, "label": label, "before": before, "after": after}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntu", type=Path)
    parser.add_argument("--urfall", type=Path)
    parser.add_argument("--gmdcsa", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ntu-falls-only", action="store_true")
    args = parser.parse_args(argv)
    cfg = Config()
    model = FallDetector(cfg, MODELS_DIR / "fall_model.joblib")._model
    if model is None:
        raise RuntimeError("Model jatuh tidak tersedia")
    rows = []

    def add(dataset, name, label, frames, height):
        before, after = compare(frames, height, model, cfg)
        rows.append({"dataset": dataset, "name": name, "label": label, "before": before, "after": after})
        if len(rows) % 250 == 0:
            print(f"{len(rows)} putaran", flush=True)

    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for dataset, path in (("urfall", args.urfall), ("gmdcsa", args.gmdcsa)):
            if path is None:
                continue
            for trace in sorted(path.glob("*.npz")):
                tr = PoseTrace.load(trace)
                if dataset == "urfall":
                    tr = hold(tr, 1.5, cfg.rate_hz)
                name = trace.stem.split(".")[0]
                label = "jatuh" if "fall" in name else ("push-up" if name in PUSH_UP else "biasa")
                add(dataset, trace.stem, label, zip(tr.t, tr.people), tr.frame_h)
    if args.ntu:
        labels = FALL if args.ntu_falls_only else (FALL | NOT_FALL)
        jobs = [(a["keypoint"], a["keypoint_score"], a["img_shape"][0], a["frame_dir"], labels[a["label"] + 1])
                for a in safe_load(args.ntu)["annotations"] if a["label"] + 1 in labels]
        with Pool(args.workers, initializer=_init_ntu) as pool:
            for row in pool.imap(_ntu_job, jobs, chunksize=32):
                rows.append(row)
                if len(rows) % 500 == 0:
                    print(f"{len(rows)} putaran", flush=True)
    counts = Counter()
    for row in rows:
        key = (row["dataset"], row["label"])
        counts[key + ("total",)] += 1
        for version in ("before", "after"):
            counts[key + (version,)] += bool(row[version])
    print(dict(sorted(counts.items())), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
