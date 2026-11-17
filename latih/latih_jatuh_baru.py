"""Latih kandidat jatuh dengan CAUCAFall, lalu ukur alarm per klip dan subjek."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from sahabat.config import Config
from sahabat.fall import FallDetector
from sahabat.history import EPS, History
from sahabat.person_features import PERSON_FEATURE_NAMES, person_features

from .fall_data import landing_time, stack, time_labels, video_rows
from .fall_train import gmdcsa_clips, ntu_clips, urfall_clips
from .putar import PoseTrace, load_config_snapshot
from .ukur_jatuh_ntu import PUSH_UP, _Predictions

CANDIDATES = ("landing10", "landing30", "descent20", "pose20")


class _RuleSupport(FallDetector):
    def __init__(self, cfg, threshold):
        super().__init__(cfg)
        self.threshold = threshold
        self.current_support = {}

    def _rule_tick(self, t, tracks, frame_h, full_rule=True):
        proposed = super()._rule_tick(t, tracks, frame_h, full_rule)
        return {person: source for person, source in proposed.items()
                if source != "aturan" or self.current_support.get(person, 1.0) >= self.threshold}


def annotation(root: Path):
    grouped = defaultdict(list)
    with (root / "caucafall-labels.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            grouped[Path(row["path"]).name].append(row)
    if len(grouped) != 100:
        raise ValueError(f"CAUCAFall: {len(grouped)} klip berlabel, seharusnya 100")
    for name, rows in sorted(grouped.items()):
        subject = int(rows[0]["subject"])
        falls = [r for r in rows if int(r["label"]) == 1]
        if name.startswith("Fall") != bool(falls) or len(falls) > 1:
            raise ValueError(f"Anotasi jatuh tidak konsisten untuk {name}")
        onset = float(falls[0]["start"]) if falls else None
        landing = float(falls[0]["end"]) if falls else None
        yield name, subject, onset, landing


def caucafall(root, cfg, subjects):
    for name, subject, onset, landing in annotation(root):
        if subject not in subjects:
            continue
        tr = PoseTrace.load(root / "cauca_416" / f"{name}.npz")
        clip = video_rows("cauca", name, subject, tr, cfg, onset, landing,
                          "jatuh" if onset is not None else name.removesuffix(f"S{subject}"))
        yield clip, landing


def legacy(data, cfg, out):
    cache = out / "fall-core.joblib"
    if cache.is_file():
        return joblib.load(cache)
    ntu = ntu_clips(data / "ntu" / "ntu120_2d.pkl", 1)
    videos = urfall_clips(data / "urfall", cfg) + gmdcsa_clips(data / "gmdcsa24" / "ekstrak", cfg)
    print("sumber lama", len(ntu), len(videos), flush=True)
    core = stack(ntu + videos, {"urfall": 10, "gmdcsa": 10})
    joblib.dump((core, videos), cache, compress=3)
    return core, videos


def model_events(clip, cfg, models, gates=None):
    """Batch prediksi, lalu gunakan penilai jatuh aplikasi dan aturan yang sama."""
    tr = clip.trace
    history = History(cfg)
    frames = list(zip(map(float, tr.t), tr.people))
    feats, sizes, track_ids = [], [], []
    for t, people in frames:
        history.update(t, people)
        start = len(feats)
        current_ids = []
        for track in history.tracks.values():
            if track.mode == "full" and track.samples and t - track.samples[-1].t <= EPS:
                vector = person_features(track, cfg.fall_window_s)
                if vector is not None:
                    feats.append(vector)
                    current_ids.append(track.track_id)
        sizes.append(len(feats) - start)
        track_ids.append(current_ids)
    prepared = np.asarray(feats).reshape(-1, len(PERSON_FEATURE_NAMES))
    probs = {name: model.predict_proba(prepared) if len(prepared) else np.empty((0, 2))
             for name, model in models.items()}
    outputs = {name: [] for name in models}
    history = History(cfg)
    gates = gates or {}
    detectors = {name: _RuleSupport(cfg, gates[name]) if gates.get(name) else FallDetector(cfg) for name in models}
    for detector in detectors.values():
        detector._model = _Predictions()
    start = 0
    for (t, people), size, ids in zip(frames, sizes, track_ids, strict=True):
        history.update(t, people)
        for name, detector in detectors.items():
            detector._model.current = probs[name][start:start + size]
            if isinstance(detector, _RuleSupport):
                detector.current_support = dict(zip(ids, probs[name][start:start + size, 1], strict=True))
            outputs[name].extend({"t": t, "source": detector.sources[person]}
                                 for person in detector.tick(t, history.tracks, tr.frame_h))
        start += size
    return outputs


def measure(clips, cfg, models, gates=None):
    details = {name: [] for name in models}
    for clip, landing in clips:
        events = model_events(clip, cfg, models, gates)
        for name, detected in events.items():
            good = [e["t"] for e in detected if clip.is_fall and e["t"] >= clip.onset - EPS
                    and (landing is None or e["t"] <= landing + 2)]
            early = [e["t"] for e in detected if clip.is_fall and e["t"] < clip.onset - EPS]
            late = any(e["t"] > landing + 2 for e in detected) if landing is not None else False
            details[name].append({"source": clip.source, "clip": clip.name, "subject": clip.group,
                                  "fall": clip.is_fall, "onset": clip.onset, "landing": landing,
                                  "scored_frames": int(np.isfinite(clip.feats).any(axis=1).sum()),
                                  "activity": "push-up" if clip.name.split(".")[0] in PUSH_UP else clip.kind,
                                  "delay_s": round(min(good) - clip.onset, 3) if good else None,
                                  "early": bool(early), "late": late, "events": detected})
    result = {}
    for name, rows in details.items():
        stats = {}
        for source in sorted({r["source"] for r in rows}):
            pos = [r for r in rows if r["source"] == source and r["fall"]]
            neg = [r for r in rows if r["source"] == source and not r["fall"]]
            delays = [r["delay_s"] for r in pos if r["delay_s"] is not None]
            push = [r for r in neg if r["activity"] == "push-up"]
            stats[source] = {"tp": len(delays), "pos": len(pos), "fp": sum(bool(r["events"]) for r in neg),
                             "neg": len(neg), "early": sum(r["early"] for r in pos),
                             "late": sum(r["late"] for r in pos),
                             "push_up_fp": sum(bool(r["events"]) for r in push),
                             "push_up": len(push),
                             "scored_clips": sum(r["scored_frames"] > 0 for r in pos + neg),
                             "median_delay": round(float(np.median(delays)), 3) if delays else None}
        result[name] = {"rule_support_threshold": (gates or {}).get(name, 0), "metrics": stats, "rows": rows}
        print(name, stats, flush=True)
    return result


def fit_candidate(core, rows, name):
    clips = []
    for clip, landing in rows:
        if name == "descent20" and clip.is_fall:
            labels = clip.labels.copy()
            # Arah dan kecepatan turun diamati lebih awal daripada pose akhir.
            # Hindari awal goyah dan hanya pilih separuh akhir transisi.
            start = (clip.onset + landing) / 2
            labels[(clip.trace.t >= start) & (clip.trace.t <= landing + 1.0)] = 1
            clip = replace(clip, labels=labels)
        if name == "pose20" and clip.is_fall:
            inferred = min(landing, landing_time(clip.trace, clip.onset))
            clip = replace(clip, labels=time_labels(clip.trace.t, clip.onset, inferred))
        clips.append(clip)
    weight = {"landing10": 10, "landing30": 30, "descent20": 20, "pose20": 20}[name]
    extra = stack(clips, {"cauca": weight})
    x, y, w = (np.concatenate((a, b)) for a, b in zip(core, extra, strict=True))
    print(name, f"{len(y)} baris, CAUCA {len(extra[1])} dengan {int(extra[1].sum())} positif", flush=True)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                          class_weight="balanced", random_state=0).fit(x, y, sample_weight=w)


def train_caucafall(core, out, cfg):
    train = list(caucafall(out, cfg, set(range(1, 7))))
    models = {}
    for name in CANDIDATES:
        file = out / f"fall-{name}.joblib"
        if file.is_file():
            models[name] = joblib.load(file)
            continue
        models[name] = fit_candidate(core, train, name)
        temporary = file.with_suffix(".part")
        joblib.dump(models[name], temporary, compress=3)
        temporary.replace(file)
        print(name, "selesai", flush=True)
    return models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("baseline", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--train-only", action="store_true")
    modes.add_argument("--baseline-dev", action="store_true")
    modes.add_argument("--gated-dev", action="store_true")
    modes.add_argument("--final", choices=("baseline", *CANDIDATES))
    parser.add_argument("--gate", type=float, default=0)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    config_file = args.out / "config-baseline.json"
    cfg = load_config_snapshot(config_file)
    core, videos = legacy(args.data, cfg, args.out)
    if args.prepare:
        return
    if args.train_only:
        train_caucafall(core, args.out, cfg)
        return
    models = {"baseline": joblib.load(args.baseline)}
    gates = {}
    if args.final:
        if args.gate:
            name = f"{args.final}-gate{args.gate:g}"
            models[name] = models["baseline"] if args.final == "baseline" else joblib.load(
                args.out / f"fall-{args.final}.joblib")
            gates[name] = args.gate
        elif args.final != "baseline":
            models[args.final] = joblib.load(args.out / f"fall-{args.final}.joblib")
        clips = list(caucafall(args.out, cfg, {9, 10}))
        clips.extend((clip, None) for clip in videos)
        report = "fall-test-gated.json" if args.gate else "fall-test.json"
    else:
        clips = list(caucafall(args.out, cfg, {7, 8}))
        report = "fall-baseline-dev.json" if args.baseline_dev else "fall-dev.json"
        if not args.baseline_dev:
            models.update(train_caucafall(core, args.out, cfg))
        if args.gated_dev:
            original = dict(models)
            models = {"baseline": models["baseline"]}
            for name, model in original.items():
                for gate in (0.05, 0.2, 0.5):
                    label = f"{name}-gate{gate:g}"
                    models[label] = model
                    gates[label] = gate
            report = "fall-dev-gated.json"
    result = measure(clips, cfg, models, gates)
    (args.out / report).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
