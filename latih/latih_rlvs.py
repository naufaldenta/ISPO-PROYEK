"""Latih dan ukur model kekerasan dengan kelompok video RLVS yang dibekukan."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np

from sahabat.config import Config
from .latih_kekerasan_baru import Replay, VARIANTS as EARLIER_VARIANTS, core, evaluate, metrics, predict, prepare
from .putar import PoseTrace, load_config_snapshot
from .train_rf import _pair_probs
from .violence_train import fit, stack

VARIANTS = ("rlvs-motion3", "rlvs1", "rlvs3", "rlvs10")
HARD_VARIANTS = ("rlvs-hard1", "rlvs-hard3", "rlvs-peak-hard1", "rlvs-peak-hard3")
EVALUATION_VARIANTS = (*EARLIER_VARIANTS, *VARIANTS, *HARD_VARIANTS, "rlvs-hard1-fall")


def items(root, part, cfg):
    manifest = json.loads((root / "rlvs-split.json").read_text(encoding="utf-8"))
    cache = root / "rlvs-features"
    cache.mkdir(exist_ok=True)
    result = []
    for row in manifest:
        if row["split"] != part:
            continue
        name = Path(row["name"]).stem
        path = cache / f"{name}.joblib"
        if path.is_file():
            result.append(joblib.load(path))
            continue
        trace = root / "rlvs_416" / f"{name}.npz"
        deadline = time.monotonic() + 3600
        while not trace.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(trace)
            time.sleep(5)
        tr = PoseTrace.load(trace)
        replay = prepare(row["name"], "rlvs", row["violent"], 7000 + row["group"], part,
                         list(zip(map(float, tr.t), tr.people)), (tr.frame_h, tr.frame_w), cfg)
        temporary = path.with_suffix(".part")
        joblib.dump(replay, temporary, compress=3)
        temporary.replace(path)
        result.append(replay)
        if len(result) % 20 == 0:
            print(part, len(result), flush=True)
    return result


def train(data, root, cfg, hard=False):
    replay = items(root, "train", cfg)
    old = core(data, cfg)
    clips = [r.item for r in replay]
    print("Latih RLVS", len(clips), "klip", sum(bool(it.clip.rows) for it in clips), "dengan fitur", flush=True)
    peaks = None
    variants = (("rlvs-motion3", 3), ("rlvs1", 1), ("rlvs3", 3), ("rlvs10", 10))
    negatives = []
    if hard:
        negatives = [r.item for r in joblib.load(root / "airtlab-replay.joblib")
                     if r.split == "train" and not r.item.violent]
        selector = joblib.load(root / "violence-rlvs-motion3.joblib")
        peaks = _pair_probs(selector, [it.clip for it in clips])
        variants = tuple((name, 3) for name in HARD_VARIANTS)
    for name, weight in variants:
        path = root / f"violence-{name}.joblib"
        if path.is_file():
            model = joblib.load(path)
        else:
            first = None if name in ("rlvs-motion3", "rlvs-hard1", "rlvs-hard3") else peaks
            sources = [old, stack(clips, weight, first)]
            if hard:
                negative_weight = 1 if name.endswith("hard1") else 3
                sources.append(stack(negatives, negative_weight))
            x, y, w = [np.concatenate(parts) for parts in zip(*sources, strict=True)]
            print(name, len(y), "baris", flush=True)
            model = fit(x, y, w)
            temporary = path.with_suffix(".part")
            joblib.dump(model, temporary, compress=3)
            temporary.replace(path)
        if name == "rlvs-motion3":
            peaks = _pair_probs(model, [it.clip for it in clips])
        print(name, "selesai", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=("train", "hard-train", "dev", "air-dev", "test"))
    parser.add_argument("--model", choices=EVALUATION_VARIANTS)
    parser.add_argument("--threshold", type=float, default=.8)
    args = parser.parse_args(argv)
    cfg = load_config_snapshot(args.root / "config-baseline.json")
    if args.stage in ("train", "hard-train"):
        train(args.data, args.root, cfg, hard=args.stage == "hard-train")
        return
    if args.stage == "air-dev":
        # Cache AIRTLab lama mereferensikan kelas Replay pada modul __main__.
        replay = [r for r in joblib.load(args.root / "airtlab-replay.joblib") if r.split == "dev"]
    else:
        replay = items(args.root, args.stage, cfg)
    models = {"baseline": joblib.load(args.root / "violence_model.joblib")}
    if args.stage in ("dev", "air-dev"):
        models.update({name: joblib.load(args.root / f"violence-{name}.joblib")
                       for name in EVALUATION_VARIANTS
                       if (args.root / f"violence-{name}.joblib").is_file()})
        rows = []
        for name, model in models.items():
            probs = predict(model, replay, cfg)
            for threshold in (.35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95):
                config = replace(cfg, violence_threshold=threshold)
                score = metrics(evaluate(replay, probs, config), config)
                rows.append({"model": name, "threshold": threshold, **score})
            print(name, "dev selesai", flush=True)
        report = "rlvs-air-dev.json" if args.stage == "air-dev" else "rlvs-dev.json"
        (args.root / report).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    else:
        if not args.model:
            parser.error("Pilih --model sebelum membuka uji akhir")
        models[args.model] = joblib.load(args.root / f"violence-{args.model}.joblib")
        result = {}
        for name, model in models.items():
            config = cfg if name == "baseline" else replace(cfg, violence_threshold=args.threshold)
            rows = evaluate(replay, predict(model, replay, config), config)
            result[name] = {"threshold": config.violence_threshold, "metrics": metrics(rows, config), "rows": rows}
            print(name, result[name]["metrics"], flush=True)
        (args.root / "rlvs-test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
