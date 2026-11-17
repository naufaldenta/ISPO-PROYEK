"""Latih kandidat pose dengan AIRTLab dan ukur alarm pada grup adegan terpisah."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np

from sahabat.alarm import AlarmManager
from sahabat.config import Config
from sahabat.engine import Engine
from sahabat.gestures import GestureDetector
from sahabat.history import History
from sahabat.violence import PairScore

from .putar import PoseTrace, load_config_snapshot
from .train_rf import _pair_probs, group_of, keep_clip, label_of, load_actions
from .violence_train import Item, fit, motion_filtered, ntu_items, stack, ut_items, webcam_items
from .windows import ClipPairs, load_clip
from sahabat.pair_features import close_pairs, pair_windows

VARIANTS = ("peak10", "peak30", "motion15", "mil10", "mil30", "regularized",
            "mix2", "mix5", "extra-trees", "core-control")

@dataclass
class Replay:
    item: Item
    split: str
    ticks: list


def prepare(name, source, violent, group, split, frames, shape, cfg):
    history = History(cfg)
    gestures = GestureDetector(cfg)
    rows, ticks = [], []
    for i, (t, people) in enumerate(frames):
        people = history.update(t, people)
        tracks = history.tracks
        modes = {track_id: tr.mode for track_id, tr in tracks.items()}
        seen = {p.track_id for p in people}
        exiting = Engine._exiting(t, tracks, seen, shape) if seen else set()
        for a, b in close_pairs(tracks, cfg.pair_max_dist):
            both = pair_windows(a, b, cfg.violence_window_s)
            if both is not None:
                rows.append((i, a.track_id, b.track_id, *both))
        ticks.append((modes, gestures.tick(t, tracks), seen, exiting))
    clip = ClipPairs([t for t, _ in frames], rows)
    return Replay(Item(source, name, violent, group, clip), split, ticks)


def airtlab(data: Path, cfg: Config) -> list[Replay]:
    root = data / "airtlab_pose"
    labels = data / "airtlab" / "violence-detection-dataset"
    actions = load_actions(labels / "violent-action-classes.csv")
    grouped = {}
    for path in sorted(root.rglob("*.npz")):
        rel = path.relative_to(root).as_posix()
        label = label_of(rel)
        if label is not None and keep_clip(label, path.stem, actions):
            grouped.setdefault((label, group_of(rel)), []).append(path)
    replay = []
    for label in (0, 1):
        groups = sorted((g for y, g in grouped if y == label),
                        key=lambda group: hashlib.sha256(f"20261116/{group}".encode()).digest())
        for i, group in enumerate(groups):
            split = "train" if i % 5 < 3 else ("dev" if i % 5 == 3 else "test")
            for path in grouped[label, group]:
                with np.load(path, allow_pickle=False) as z:
                    shape = tuple(map(int, z["frame_hw"]))
                replay.append(prepare(path.relative_to(root).as_posix(), "airtlab", bool(label),
                                      3000 + label * 1000 + i, split, load_clip(path), shape, cfg))
    return replay


def bus(root: Path, cfg: Config) -> list[Replay]:
    cache = root / "bus-replay.joblib"
    if cache.is_file():
        return joblib.load(cache)
    from zipfile import ZipFile
    with ZipFile(root / "bus-violence.zip") as archive:
        names = archive.read("test.txt").decode().splitlines()
    replay = []
    for i, name in enumerate(names):
        tr = PoseTrace.load(root / "bus_416" / (Path(name).stem + ".npz"))
        replay.append(prepare(name, "bus", name.startswith("VIOLENCE_"), 5000 + i, "test",
                              list(zip(map(float, tr.t), tr.people)), (tr.frame_h, tr.frame_w), cfg))
    temporary = cache.with_suffix(".part")
    joblib.dump(replay, temporary, compress=3)
    temporary.replace(cache)
    return replay


def predict(model, replay, cfg):
    items = [r.item for r in replay]
    return motion_filtered(items, _pair_probs(model, [it.clip for it in items]), cfg)


def evaluate(replay, probs, cfg):
    rows = []
    for r, p in zip(replay, probs, strict=True):
        clip = r.item.clip
        windows = {}
        for (index, a, b, _, _), value in zip(clip.rows, p, strict=True):
            windows.setdefault(index, []).append(PairScore((a, b), float(value)))
        manager = AlarmManager(cfg)
        events = []
        for i, (t, (modes, hits, seen, exiting)) in enumerate(zip(clip.t, r.ticks, strict=True)):
            events.extend(manager.tick(t, [], windows.get(i, []), modes, hits,
                                       seen=seen if cfg.gesture_enabled else None, exiting=exiting))
        pairs = {}
        for scores in windows.values():
            for score in scores:
                pairs[score.ids] = pairs.get(score.ids, 0) + 1
        rows.append({"name": r.item.name, "violent": r.item.violent, "group": r.item.group,
                     "split": r.split, "scored": bool(clip.rows),
                     "pair_ticks": max(pairs.values(), default=0),
                     "events": [{"t": round(e.t, 3), "source": e.source} for e in events]})
    return rows


def metrics(rows, cfg):
    pos = [row for row in rows if row["violent"]]
    neg = [row for row in rows if not row["violent"]]
    return {"tp": sum(bool(row["events"]) for row in pos), "pos": len(pos),
            "fp": sum(bool(row["events"]) for row in neg), "neg": len(neg),
            "scored": sum(row["scored"] for row in rows),
            "enough_pair_ticks": sum(row["pair_ticks"] >= cfg.confirm_k for row in rows),
            "scored_pos": sum(row["scored"] for row in pos),
            "scored_neg": sum(row["scored"] for row in neg),
            "enough_pair_ticks_pos": sum(row["pair_ticks"] >= cfg.confirm_k for row in pos),
            "enough_pair_ticks_neg": sum(row["pair_ticks"] >= cfg.confirm_k for row in neg)}


def core(data, cfg):
    cache = data / "putaran3" / "violence-core.joblib"
    if cache.is_file():
        print("Memuat cache baris latih lama", flush=True)
        return joblib.load(cache)
    ntu = data / "ntu"
    items = ntu_items(ntu / "pairs_cache_d4.npz")
    print("Core NTU", len(items), flush=True)
    items += ut_items(data / "utinteraction" / "ekstrak", cfg)
    print("Core + UT", len(items), flush=True)
    items += webcam_items(data / "webcam" / "sesi-20261115-211259", cfg)
    print("Core + webcam", len(items), flush=True)
    with np.load(ntu / "oof_kekerasan.npz", allow_pickle=False) as z:
        offsets = z["offsets"]
        if list(np.diff(offsets)) != [len(it.clip.rows) for it in items]:
            raise ValueError("Urutan cache peluang NTU/UT/webcam berbeda dari baris latih")
        first = np.split(z["first"], offsets[1:-1])
    rows = stack(items, 10, first)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".part")
    joblib.dump(rows, temporary, compress=3)
    temporary.replace(cache)
    print("Core tersimpan", len(rows[1]), flush=True)
    return rows


def fit_candidates(data, out, incumbent, replay, cfg):
    train = [r for r in replay if r.split == "train"]
    dev = [r for r in replay if r.split == "dev"]
    # Untuk klip positif berlabel per-video, pakai puncak model lama atau semua
    # jendela bergerak. Negatif dipakai seluruhnya agar pembanding adil.
    items = [r.item for r in train]
    peaks = _pair_probs(incumbent, [i.clip for i in items])
    core_rows = core(data, cfg)
    report = {}
    base = evaluate(dev, predict(incumbent, dev, cfg), cfg)
    report["baseline"] = {"dev": metrics(base, cfg), "rows": base}
    print("baseline", report["baseline"]["dev"], flush=True)
    for name, weight, first in (("peak10", 10, peaks), ("peak30", 30, peaks),
                                ("motion15", 15, None)):
        file = out / f"violence-{name}.joblib"
        if file.exists():
            model = joblib.load(file)
        else:
            extra = stack(items, weight, first)
            combined = [np.concatenate((a, b)) for a, b in zip(core_rows, extra, strict=True)]
            print(name, "baris latih", len(combined[1]), "tambahan", len(extra[1]), flush=True)
            model = fit(*combined)
            joblib.dump(model, file, compress=3)
        rows = evaluate(dev, predict(model, dev, cfg), cfg)
        report[name] = {"dev": metrics(rows, cfg), "rows": rows}
        print(name, report[name]["dev"], flush=True)
        (out / "violence-dev.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def sweep(out, baseline, replay, cfg):
    dev = [r for r in replay if r.split == "dev"]
    scores = []
    models = {"baseline": baseline}
    for name in VARIANTS:
        file = out / f"violence-{name}.joblib"
        if file.is_file():
            models[name] = joblib.load(file)
    for name, model in models.items():
        probs = predict(model, dev, cfg)
        for threshold in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            settings = replace(cfg, violence_threshold=threshold)
            metrics_ = metrics(evaluate(dev, probs, settings), settings)
            scores.append({"model": name, "threshold": threshold, **metrics_})
    (out / "violence-threshold-dev.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    for score in scores:
        if score["tp"] >= 30 or score["fp"] <= 2:
            print(score, flush=True)


def second_round(data, out, replay, cfg):
    from sklearn.ensemble import HistGradientBoostingClassifier
    items = [r.item for r in replay if r.split == "train"]
    dev = [r for r in replay if r.split == "dev"]
    selector = joblib.load(out / "violence-motion15.joblib")
    peaks = _pair_probs(selector, [it.clip for it in items])
    core_rows = core(data, cfg)
    report = json.loads((out / "violence-dev.json").read_text(encoding="utf-8"))
    for name, weight in (("mil10", 10), ("mil30", 30), ("regularized", 20)):
        file = out / f"violence-{name}.joblib"
        if file.exists():
            model = joblib.load(file)
        else:
            extra = stack(items, weight, peaks)
            x, y, w = [np.concatenate((a, b)) for a, b in zip(core_rows, extra, strict=True)]
            print(name, "baris", len(y), flush=True)
            if name == "regularized":
                model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=15,
                                                        l2_regularization=10, class_weight="balanced",
                                                        random_state=0).fit(x, y, sample_weight=w)
            else:
                model = fit(x, y, w)
            temporary = file.with_suffix(".part")
            joblib.dump(model, temporary, compress=3)
            temporary.replace(file)
        rows = evaluate(dev, predict(model, dev, cfg), cfg)
        report[name] = {"dev": metrics(rows, cfg), "rows": rows}
        print(name, report[name]["dev"], flush=True)
        (out / "violence-dev.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def third_round(data, out, replay, cfg, only_control=False):
    from sklearn.ensemble import ExtraTreesClassifier
    items = [r.item for r in replay if r.split == "train"]
    dev = [r for r in replay if r.split == "dev"]
    selector = joblib.load(out / "violence-motion15.joblib")
    peaks = _pair_probs(selector, [it.clip for it in items])
    core_rows = core(data, cfg)
    report = json.loads((out / "violence-dev.json").read_text(encoding="utf-8"))
    variants = (("core-control", None),) if only_control else (("mix2", 2), ("mix5", 5), ("extra-trees", 5))
    for name, weight in variants:
        file = out / f"violence-{name}.joblib"
        if file.exists():
            model = joblib.load(file)
        else:
            if weight is None:
                x, y, w = core_rows
            else:
                extra = stack(items, weight, peaks)
                x, y, w = [np.concatenate((a, b)) for a, b in zip(core_rows, extra, strict=True)]
            print(name, "baris", len(y), flush=True)
            if name == "extra-trees":
                model = ExtraTreesClassifier(n_estimators=128, max_depth=14, min_samples_leaf=10,
                                              class_weight="balanced", random_state=0, n_jobs=2)
                model.fit(x, y, sample_weight=w)
                model.n_jobs = 1
            else:
                model = fit(x, y, w)
            temporary = file.with_suffix(".part")
            joblib.dump(model, temporary, compress=3)
            temporary.replace(file)
        rows = evaluate(dev, predict(model, dev, cfg), cfg)
        report[name] = {"dev": metrics(rows, cfg), "rows": rows}
        print(name, report[name]["dev"], flush=True)
        (out / "violence-dev.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--final", choices=("baseline", *VARIANTS))
    parser.add_argument("--bus", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--round2", action="store_true")
    parser.add_argument("--round3", action="store_true")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--threshold", type=float, default=Config().violence_threshold)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    config_file = args.out / "config-baseline.json"
    cfg = load_config_snapshot(config_file)
    selected_cfg = replace(cfg, violence_threshold=args.threshold)
    if not 0 < args.threshold < 1:
        parser.error("--threshold harus lebih dari 0 dan kurang dari 1")
    model = joblib.load(args.baseline)
    cache = args.out / "airtlab-replay.joblib"
    if cache.is_file():
        replay = joblib.load(cache)
    else:
        replay = airtlab(args.data, cfg)
        joblib.dump(replay, cache, compress=3)
    print({part: len([r for r in replay if r.split == part]) for part in ("train", "dev", "test")}, flush=True)
    if args.control:
        third_round(args.data, args.out, replay, cfg, only_control=True)
    elif args.round3:
        third_round(args.data, args.out, replay, cfg)
    elif args.round2:
        second_round(args.data, args.out, replay, cfg)
    elif args.sweep:
        sweep(args.out, model, replay, cfg)
    elif args.final or args.bus:
        test = bus(args.data / "putaran3", cfg) if args.bus else [r for r in replay if r.split == "test"]
        models = {"baseline": model}
        if args.final and args.final != "baseline":
            models[args.final] = joblib.load(args.out / f"violence-{args.final}.joblib")
        result = {}
        for name, candidate in models.items():
            settings = cfg if name == "baseline" else selected_cfg
            rows = evaluate(test, predict(candidate, test, settings), settings)
            result[name] = {"threshold": settings.violence_threshold, "metrics": metrics(rows, settings), "rows": rows}
            print(name, result[name]["metrics"], flush=True)
        (args.out / ("violence-bus.json" if args.bus else "violence-test.json")).write_text(
            json.dumps(result, indent=2), encoding="utf-8")
    else:
        fit_candidates(args.data, args.out, model, replay, cfg)


if __name__ == "__main__":
    main()
