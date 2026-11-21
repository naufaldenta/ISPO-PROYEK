"""Model kekerasan untuk kamera dekat (setengah badan, agak menghadap kamera, komputer kadang hanya 5 penilaian/detik).

    python -m latih.latih_dekat siapkan     putar ulang sesi webcam pada laju asli dan setengahnya (disimpan di cache)
    python -m latih.latih_dekat uji         uji lintas sesi: dilatih TANPA sesi yang diuji, sapu ambang dan konfirmasi

Melatih model dari rekaman kamera lomba dan memasangnya: latih.kalibrasi_dekat.

Semua baris dilihat tanpa kaki (`tanpa_kaki`), jadi orang yang kakinya terpotong dan yang terlihat utuh dinilai sama.
Sumber: baris latih model terpasang (NTU punya tiga sudut kamera, termasuk menghadap dan serong) ditambah sesi 16 dan 17
November pada dua laju. Adegan cekik/menjepit tidak dipakai sebagai positif karena ditangani aturan kuncian.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np

from sahabat.alarm import AlarmManager
from sahabat.pair_features import MOTION_FEATURES, tanpa_kaki
from sahabat.violence import PairScore

from .latih_kekerasan_baru import Replay, prepare  # noqa: F401  (Replay dibutuhkan untuk memuat cache)
from .latih_negatif_jatuh import installed_sources
from .latih_sesi_jauh import IGNORE_AFTER, IGNORE_BEFORE, REAL_GAP_ALARMS, START, segments
from .putar import PoseTrace, load_config_snapshot
from .violence_train import fit

DATA = Path("D:/sahabat-data")
P3 = DATA / "putaran3"
OUT = DATA / "dekat"
SESI = {"s16": DATA / "webcam" / "sesi-20261116-153538", "s17": DATA / "webcam" / "sesi-20261117-101936"}
STEPS = (1, 2)
KUNCIAN = {"K4", "K5", "K9"}
PUKULAN = {"K1", "K6", "K8"}
ACK_AFTER_S = 3.0
GRACE_S = 3.0
SESI_BOBOT = 10.0
AMBANG = (0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
KONFIRMASI = ((1.0, 0.6, 3), (1.0, 0.5, 3), (0.8, 0.6, 3), (0.6, 0.6, 2), (1.0, 0.4, 2))


def _replay(name: str, step: int, cfg) -> Replay:
    cache = OUT / f"replay-{name}-{step}.joblib"
    if cache.is_file():
        return joblib.load(cache)
    tr = PoseTrace.load(SESI[name] / "jejak_416.npz")
    frames = list(zip(map(float, np.asarray(tr.t)[::step]), list(tr.people)[::step]))
    replay = prepare(name, "webcam", False, 0, "sesi", frames, (tr.frame_h, tr.frame_w), cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(replay, cache, compress=3)
    return replay


def _scene(text: str) -> str:
    return text.split()[0].rstrip(";")


def labels(name: str) -> dict:
    """Rentang latih dan rentang uji satu sesi, dalam detik jejak."""
    tr = PoseTrace.load(SESI[name] / "jejak_416.npz")
    end = float(tr.t[-1])
    if name == "s16":
        t0 = float(tr.t[0])
        ranges = json.loads((SESI[name] / "label.json").read_text(encoding="utf-8"))
        takes = [{"t0": t0 + r["mulai"], "t1": t0 + r["selesai"], "kekerasan": bool(r["kekerasan"]),
                  "adegan": _scene(r.get("catatan", ""))} for r in ranges]
        train = [{"t0": k["t0"], "t1": k["t1"], "kekerasan": k["kekerasan"]} for k in takes
                 if k["adegan"] not in KUNCIAN]
        return {"takes": takes, "train": train, "start": min(k["t0"] for k in takes), "end": end, "ignore": []}
    u = json.loads((SESI[name] / "ulangan.json").read_text(encoding="utf-8"))
    takes = [{"t0": k["lo"], "t1": k["hi"], "kekerasan": k["label"] == "kekerasan", "adegan": k["scene"]}
             for k in sorted(u["takes"], key=lambda k: k["lo"])]
    train = [{"t0": s["t0"], "t1": s["t1"], "kekerasan": s["kekerasan"]} for s in segments(u, "tanpa-kuncian")]
    ignore = [(t - IGNORE_BEFORE, t + IGNORE_AFTER) for t in REAL_GAP_ALARMS]
    return {"takes": takes, "train": train, "start": START, "end": end, "ignore": ignore}


def _matrix(replay: Replay) -> tuple[np.ndarray, np.ndarray]:
    """(fitur tanpa kaki [baris, 2, ciri], waktu tiap baris)."""
    clip = replay.item.clip
    x = np.array([(ab, ba) for _, _, _, ab, ba in clip.rows], dtype=np.float32)
    t = np.array([clip.t[f] for f, *_ in clip.rows])
    return tanpa_kaki(x), t


def session_rows(replay: Replay, segs: list[dict], weight: float):
    x, t = _matrix(replay)
    X, y = [], []
    for s in segs:
        m = (t >= s["t0"]) & (t <= s["t1"])
        if not m.any():
            continue
        rows = x[m].reshape(-1, x.shape[2])
        if s["kekerasan"]:
            rows = rows[rows[:, MOTION_FEATURES].max(axis=1) >= 1.0]
        X.append(rows)
        y.append(np.full(len(rows), int(s["kekerasan"])))
    X, y = np.concatenate(X), np.concatenate(y)
    return X, y, np.full(len(y), weight)


def base_rows(cfg):
    cache = OUT / "dasar-tanpa-kaki.joblib"
    if cache.is_file():
        return joblib.load(cache)
    x, y, w = [np.concatenate(parts) for parts in zip(*installed_sources(DATA, P3, cfg), strict=True)]
    rows = (tanpa_kaki(x), y, w)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(rows, cache)
    return rows


def probs(model, replay: Replay, cfg, masked: bool = True) -> np.ndarray:
    clip = replay.item.clip
    if masked:
        x, _ = _matrix(replay)
    else:
        x = np.array([(ab, ba) for _, _, _, ab, ba in clip.rows], dtype=np.float32)
    p = model.predict_proba(x.reshape(-1, x.shape[2]))[:, 1].reshape(-1, 2).max(axis=1)
    active = x[:, :, MOTION_FEATURES].max(axis=(1, 2)) >= cfg.violence_min_motion
    return np.where(active, p, 0.0)


def events(replay: Replay, p: np.ndarray, cfg) -> list[dict]:
    clip = replay.item.clip
    by_tick: dict[int, list[PairScore]] = {}
    for (i, a, b, _, _), v in zip(clip.rows, p, strict=True):
        by_tick.setdefault(i, []).append(PairScore((a, b), float(v)))
    manager, out, red = AlarmManager(cfg), [], None
    for i, (t, (modes, hits, seen, exiting)) in enumerate(zip(clip.t, replay.ticks, strict=True)):
        out += [{"t": e.t, "sumber": e.source}
                for e in manager.tick(t, [], by_tick.get(i, []), modes, hits, seen=seen, exiting=exiting)]
        if manager.active is not None:
            red = t if red is None else red
            if t - red >= ACK_AFTER_S:
                manager.acknowledge(t)
                red = None
        else:
            red = None
    return out


def score(evs: list[dict], lab: dict) -> dict:
    out = {"pukulan": 0, "n_pukulan": 0, "kekerasan": 0, "n_kekerasan": 0, "biasa": 0, "n_biasa": 0}
    for k in lab["takes"]:
        hit = any(k["t0"] <= e["t"] <= k["t1"] + GRACE_S for e in evs)
        if k["kekerasan"]:
            out["kekerasan"] += hit
            out["n_kekerasan"] += 1
            if k["adegan"] in PUKULAN:
                out["pukulan"] += hit
                out["n_pukulan"] += 1
        else:
            out["biasa"] += hit
            out["n_biasa"] += 1
    bad = [(k["t0"], k["t1"] + GRACE_S) for k in lab["takes"] if k["kekerasan"]] + lab["ignore"]
    grid = np.arange(lab["start"], lab["end"], 0.1)
    normal = np.ones(len(grid), dtype=bool)
    for a, b in bad:
        normal &= ~((grid >= a) & (grid <= b))
    hours = normal.sum() * 0.1 / 3600
    false = sum(lab["start"] <= e["t"] <= lab["end"]
                and not any(a <= e["t"] <= b for a, b in bad) for e in evs)
    out["keliru_per_jam"] = round(false / hours, 1)
    return out


def _variants(cfg):
    yield "terpasang-aturan", cfg
    for th in AMBANG:
        for w, frac, low in KONFIRMASI:
            yield f"a{th:g}-w{w:g}-f{frac:g}-m{low}", replace(cfg, violence_threshold=th, confirm_window_s=w,
                                                              confirm_frac=frac, confirm_min_hits=low)


def uji(cfg) -> dict:
    base = base_rows(cfg)
    installed = joblib.load(Path("models") / "violence_model.joblib")
    report = {}
    for held, other in (("s16", "s17"), ("s17", "s16")):
        lab_other = labels(other)
        extra = [session_rows(_replay(other, step, cfg), lab_other["train"], SESI_BOBOT) for step in STEPS]
        X, y, w = [np.concatenate(parts) for parts in zip(base, *extra, strict=True)]
        print(f"latih tanpa {held}: {len(y)} baris", flush=True)
        model = fit(X, y, w)
        joblib.dump(model, OUT / f"model-tanpa-{held}.joblib", compress=3)
        lab = labels(held)
        for step in STEPS:
            rep = _replay(held, step, cfg)
            hz = (len(rep.item.clip.t) - 1) / (rep.item.clip.t[-1] - rep.item.clip.t[0])
            old = score(events(rep, probs(installed, rep, cfg, masked=False), cfg), lab)
            report[f"{held}-langkah{step}-terpasang"] = {"hz": round(hz, 1), **old}
            print(f"{held} {hz:.1f} Hz terpasang", old, flush=True)
            p = probs(model, rep, cfg)
            for name, c in _variants(cfg):
                report[f"{held}-langkah{step}-{name}"] = {"hz": round(hz, 1), **score(events(rep, p, c), lab)}
            (OUT / "uji.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def ringkas(report: dict) -> None:
    names = sorted({k.split("-", 2)[2] for k in report if "terpasang" not in k.split("-", 2)[2]})
    rows = []
    for n in names:
        cells = [report[f"{h}-langkah{s}-{n}"] for h in ("s16", "s17") for s in STEPS]
        rows.append((sum(c["pukulan"] for c in cells), -sum(c["keliru_per_jam"] for c in cells), n, cells))
    def fmt(c):
        return f"{c['pukulan']}/{c['n_pukulan']} K{c['kekerasan']}/{c['n_kekerasan']} B{c['biasa']}/{c['n_biasa']} {c['keliru_per_jam']}/j"
    for h in ("s16", "s17"):
        for s in STEPS:
            c = report[f"{h}-langkah{s}-terpasang"]
            print(f"terpasang {h} {c['hz']} Hz:", fmt(c))
    for _, _, n, cells in sorted(rows, key=lambda r: r[:2], reverse=True)[:18]:
        print(f"{n:24s}", " | ".join(fmt(c) for c in cells))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tahap", choices=("siapkan", "uji", "ringkas"))
    args = ap.parse_args(argv)
    cfg = load_config_snapshot(P3 / "config-baseline.json")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.tahap == "siapkan":
        base_rows(cfg)
        for name in SESI:
            for step in STEPS:
                r = _replay(name, step, cfg)
                print(name, step, len(r.item.clip.t), "putaran", len(r.item.clip.rows), "baris", flush=True)
    elif args.tahap == "uji":
        ringkas(uji(cfg))
    else:
        ringkas(json.loads((OUT / "uji.json").read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
