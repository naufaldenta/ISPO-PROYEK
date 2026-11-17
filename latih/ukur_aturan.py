"""Ukur pengenal gerak pada NTU, UT-Interaction, dan jejak webcam berlabel."""
from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from sahabat.config import Config
from sahabat.gestures import GestureDetector
from sahabat.history import History
from .ntu import FRIENDLY, VIOLENT, safe_load, to_frames
from .putar import PoseTrace, _overrides


def clip_hits(frames, cfg: Config):
    history, detector = History(cfg), GestureDetector(cfg)
    return [(t, hit) for t, people in frames
            for hit in _step(t, people, history, detector)]


def _step(t, people, history, detector):
    history.update(t, people)
    return detector.tick(t, history.tracks)


def _ntu_job(job):
    kp, conf, action, cfg = job
    hits = clip_hits(to_frames(kp, conf), cfg)
    return action, sorted({h.kind for _, h in hits})


def summarize(rows):
    groups = defaultdict(lambda: {"total": 0, "alarm": 0})
    for label, kinds in rows:
        group = groups[label]
        group["total"] += 1
        group["alarm"] += bool(kinds)
        for kind in kinds:
            group[kind] = group.get(kind, 0) + 1
    return dict(groups)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntu", type=Path)
    parser.add_argument("--ut", type=Path)
    parser.add_argument("--webcam", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = dataclasses.replace(Config(), **_overrides(args.set))
    result = {}
    if args.ut:
        from .utinteraction import build_traces, load_index
        labels = {clip.name: clip.label for clip in load_index(args.ut)}
        traces = build_traces(args.ut, 416, cfg.rate_hz)
        result["ut"] = summarize((labels[name], {h.kind for _, h in clip_hits(zip(tr.t, tr.people), cfg)})
                                 for (name, _), tr in traces.items())
        print("UT:", result["ut"], flush=True)
    if args.webcam:
        ranges = json.loads((args.webcam / "label.json").read_text(encoding="utf-8"))
        tr = PoseTrace.load(args.webcam / "jejak_416.npz")
        hits = [(round(float(t - tr.t[0]), 3), h.kind, h.ids)
                for t, h in clip_hits(zip(tr.t, tr.people), cfg)]
        result["webcam_hits"] = hits
        result["webcam"] = summarize(
            ("kekerasan" if r["kekerasan"] else "biasa",
             {kind for t, kind, _ in hits if r["mulai"] <= t < r["selesai"]}) for r in ranges)
        print("Webcam:", result["webcam"], hits, flush=True)
    if args.ntu:
        # Pertahankan urutan klip agar bisa digabung dengan hasil model per pemeran.
        annotations = safe_load(args.ntu)["annotations"]
        jobs = [(a["keypoint"], a["keypoint_score"], a["label"] + 1, cfg)
                for a in annotations if a["label"] + 1 in (FRIENDLY | VIOLENT)]
        del annotations
        with Pool(args.workers) as pool:
            rows = []
            for row in pool.imap(_ntu_job, jobs, chunksize=32):
                rows.append(row)
                if len(rows) % 2000 == 0:
                    print(f"NTU {len(rows)}/{len(jobs)}", flush=True)
        result["ntu"] = summarize(rows)
        result["ntu_rows"] = rows
        print("NTU:", result["ntu"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
