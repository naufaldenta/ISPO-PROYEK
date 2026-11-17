"""Latih ulang model kekerasan dengan sesi kamera 3-4 m (17 November), lalu ukur di semua gerbang.

    python -m latih.latih_sesi_jauh silang     validasi silang 3 lipatan pada sesi: varian label x bobot x ambang
    python -m latih.latih_sesi_jauh kandidat --varian semua --bobot 3
                                               latih dengan seluruh sesi, ukur semua gerbang regresi pada beberapa ambang

Label dibangun dari `ulangan.json` (tanda tombol per ulangan):
- Ulangan kekerasan: positif hanya 7 detik sebelum tanda sampai 0,5 detik sesudahnya (tombol ditekan setelah adegan);
  sisa jendela ulangan diabaikan karena berisi persiapan.
- Ulangan biasa (B1-B7), sela antar-ulangan, dan kegiatan bebas: negatif.
- Diabaikan: semua sebelum detik 2030 (pengaturan kamera dan peragaan gerak), dan 8 detik sebelum hingga 3 detik sesudah
  alarm di sela yang menurut gambar memang gerak kekerasan (latihan atau ulangan yang diulang tanpa tanda).
Lipatan silang per satuan (satu ulangan beserta sela sebelumnya; kegiatan bebas dan sela awal per 60 detik).
Model kandidat tidak dipasang oleh skrip ini.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np

from sahabat.violence import PairScore  # noqa: F401  (dipakai session_events)

from .latih_kekerasan_baru import Replay, bus, evaluate, metrics, predict, prepare  # Replay: cache lama memakai __main__
from .latih_negatif_jatuh import installed_sources
from .latih_rlvs import items as rlvs_items
from .latih_sesi_webcam import OLD_SESSION, SESSION as SESSION16, score_ranges, session_events, session_replay
from .putar import PoseTrace, load_config_snapshot
from .train_rf import _pair_probs
from .violence_train import Item, ClipPairs, alarms, fit, motion_filtered, ntu_items, stack

DATA = Path("D:/sahabat-data")
OUT = DATA / "putaran3"
SESSION = DATA / "webcam" / "sesi-20261117-101936"
KELUARAN = OUT / "sesi17"
FOLDS = 3
START = 2030.0                      # sebelum ini: pengaturan kamera, orang dekat lensa, peragaan gerak
POS_BEFORE, POS_AFTER = 7.0, 0.5    # positif relatif terhadap tanda K
# Alarm model terpasang di sela yang menurut gambar memang gerak kekerasan (latihan, ulangan diulang tanpa tanda),
# atau peragaan dekat kamera. Diabaikan untuk latih maupun hitungan alarm keliru.
REAL_GAP_ALARMS = (2182.3, 2185.3, 2330.8, 2347.6, 2664.6, 3003.5, 3183.1, 3291.3, 3369.0)
IGNORE_BEFORE, IGNORE_AFTER = 8.0, 3.0
KUNCIAN = {"K4", "K5", "K9"}
WEBCAM_GROUP = 3000


def _subtract(t0: float, t1: float, holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    parts = [(t0, t1)]
    for h0, h1 in holes:
        parts = [p for a, b in parts for p in ((a, min(b, h0)), (max(a, h1), b)) if p[1] - p[0] > 0.2]
    return parts


def segments(u: dict, variant: str) -> list[dict]:
    """Rentang latih {t0, t1, kekerasan, satuan} dalam detik jejak (bukan relatif)."""
    holes = [(t - IGNORE_BEFORE, t + IGNORE_AFTER) for t in REAL_GAP_ALARMS]
    out, unit, cursor = [], 0, START
    takes = sorted(u["takes"], key=lambda k: k["lo"])
    first_lo = takes[0]["lo"]
    # sela awal sebelum K1, per 60 detik
    for a in np.arange(START, first_lo, 60.0):
        for p in _subtract(a, min(a + 60.0, first_lo), holes):
            out.append({"t0": p[0], "t1": p[1], "kekerasan": False, "satuan": unit})
        unit += 1
    cursor = first_lo
    for k in takes:
        for p in _subtract(cursor, k["lo"], holes):  # sela sebelum ulangan ini
            out.append({"t0": p[0], "t1": p[1], "kekerasan": False, "satuan": unit})
        if k["label"] == "kekerasan":
            if not (variant == "tanpa-kuncian" and k["scene"] in KUNCIAN):
                out.append({"t0": max(k["lo"], k["start"] - POS_BEFORE), "t1": k["end"] + POS_AFTER,
                            "kekerasan": True, "satuan": unit})
        else:
            for p in _subtract(k["lo"], k["hi"], holes):
                out.append({"t0": p[0], "t1": p[1], "kekerasan": False, "satuan": unit})
        cursor = max(cursor, k["hi"])
        unit += 1
    free_lo = min(f["lo"] for f in u["free"])
    for p in _subtract(cursor, free_lo, holes):
        out.append({"t0": p[0], "t1": p[1], "kekerasan": False, "satuan": unit})
    unit += 1
    for a in np.arange(free_lo, u["END"], 60.0):
        out.append({"t0": a, "t1": min(a + 60.0, u["END"]), "kekerasan": False, "satuan": unit})
        unit += 1
    return out


def items_from_replay(replay: Replay, segs: list[dict]) -> list[Item]:
    clip = replay.item.clip
    t = np.asarray(clip.t, dtype=float)
    by_frame: dict[int, list] = {}
    for row in clip.rows:
        by_frame.setdefault(row[0], []).append(row)
    items = []
    for s in segs:
        inside = np.flatnonzero((t >= s["t0"]) & (t <= s["t1"]))
        place = {int(f): j for j, f in enumerate(inside)}
        rows = [(place[f], a, b, ab, ba) for f in place for (_, a, b, ab, ba) in by_frame.get(f, [])]
        items.append(Item("webcam", "kekerasan" if s["kekerasan"] else "biasa", s["kekerasan"],
                          WEBCAM_GROUP + s["satuan"], ClipPairs([float(t[i]) for i in inside], rows)))
    return items


def scoring_ranges(u: dict) -> list[dict]:
    """Rentang uji seperti laporan sesi: tiap ulangan K/B, t0..t1 (+3 dtk di score_ranges)."""
    takes = sorted(u["takes"], key=lambda k: k["lo"])
    return [{"t0": k["lo"], "t1": k["hi"], "kekerasan": k["label"] == "kekerasan", "grup": n,
             "catatan": k["scene"]} for n, k in enumerate(takes)]


def unit_of_take(u: dict) -> list[int]:
    """Nomor satuan tiap ulangan, sama urutan dengan scoring_ranges."""
    first_lo = min(k["lo"] for k in u["takes"])
    n_lead = len(np.arange(START, first_lo, 60.0))
    return [n_lead + n for n in range(len(u["takes"]))]


def gap_alarms(u: dict, events: list[dict], units: set[int] | None, segs: list[dict]) -> int:
    """Alarm di sela dan kegiatan bebas (di luar jendela ulangan + 3 dtk), tidak di zona abaikan."""
    takes = [(k["lo"], k["hi"] + 3.0) for k in u["takes"]]
    holes = [(t - IGNORE_BEFORE, t + IGNORE_AFTER) for t in REAL_GAP_ALARMS]
    n = 0
    for e in events:
        t = e["t"]
        if t < START or any(a <= t <= b for a, b in takes) or any(a <= t <= b for a, b in holes):
            continue
        if units is not None:
            owner = [s["satuan"] for s in segs if s["t0"] - 3 <= t <= s["t1"] + 3]
            if not owner or owner[0] not in units:
                continue
        n += 1
    return n


def load_session(cfg):
    u = json.loads((SESSION / "ulangan.json").read_text(encoding="utf-8"))
    cache = KELUARAN / "sesi17-replay.joblib"
    if cache.is_file():
        replay = joblib.load(cache)
    else:
        tr = PoseTrace.load(SESSION / "jejak_416.npz")
        replay = prepare(SESSION.name, "webcam", False, 0, "sesi", list(zip(map(float, tr.t), tr.people)),
                         (tr.frame_h, tr.frame_w), cfg)
        joblib.dump(replay, cache, compress=3)
    return u, replay


def session_score(u, replay, probs, cfg, units=None, segs=None) -> dict:
    ranges = scoring_ranges(u)
    events = session_events(replay, probs, cfg)
    take_units = unit_of_take(u)
    only = None if units is None else {r["grup"] for r, un in zip(ranges, take_units) if un in units}
    s = score_ranges(ranges, events, only)
    s["sela"] = gap_alarms(u, events, units, segs)
    return s


def fit_with(base: list[tuple], extra: list[tuple]):
    x, y, w = [np.concatenate(parts) for parts in zip(*(base + extra), strict=True)]
    print(f"{len(y)} baris latih ({int(y.sum())} positif)", flush=True)
    return fit(x, y, w)


def crossval(variants, weights, thresholds):
    cfg0 = load_config_snapshot(OUT / "config-baseline.json")
    u, replay = load_session(cfg0)
    base = installed_sources(DATA, OUT, cfg0)
    installed = joblib.load(OUT / "violence-selected.joblib")
    report = {}
    p_inst = predict(installed, [replay], cfg0)[0]
    for th in thresholds:
        cfg = replace(cfg0, violence_threshold=th)
        s = session_score(u, replay, p_inst, cfg, None, segments(u, "semua"))
        report[f"terpasang-ambang{th:g}"] = {k: s[k] for k in ("tp", "pos", "fp", "neg", "sela")}
        print(f"terpasang ambang {th:g}", report[f"terpasang-ambang{th:g}"], flush=True)
    for variant in variants:
        segs = segments(u, variant)
        items = items_from_replay(replay, segs)
        all_units = sorted({s["satuan"] for s in segs})
        for w in weights:
            folds = []
            for fold in range(FOLDS):
                test = {un for un in all_units if un % FOLDS == fold}
                train_items = [it for it, s in zip(items, segs, strict=True) if s["satuan"] not in test]
                model = fit_with(base, [stack(train_items, w)])
                folds.append((test, predict(model, [replay], cfg0)[0]))
            for th in thresholds:
                cfg = replace(cfg0, violence_threshold=th)
                merged = {"tp": 0, "pos": 0, "fp": 0, "neg": 0, "sela": 0}
                for test, p in folds:
                    s = session_score(u, replay, p, cfg, test, segs)
                    for k in merged:
                        merged[k] += s[k]
                key = f"{variant}-bobot{w:g}-ambang{th:g}"
                report[key] = merged
                print(key, merged, flush=True)
            (KELUARAN / "silang.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def all_gates(model, thresholds) -> dict:
    """Semua gerbang regresi (RLVS, AIRTLab, Bus, UT, NTU, sesi 15, sesi 16) pada beberapa ambang."""
    cfg0 = load_config_snapshot(OUT / "config-baseline.json")
    sets = {"rlvs": rlvs_items(OUT, "test", cfg0),
            "airtlab": [r for r in joblib.load(OUT / "airtlab-replay.joblib") if r.split == "test"],
            "bus": bus(OUT, cfg0), "ut": joblib.load(OUT / "ut-replay.joblib")}
    probs = {name: predict(model, rep, cfg0) for name, rep in sets.items()}
    ntu = ntu_items(DATA / "ntu" / "pairs_cache_d4.npz")
    violent = np.array([it.violent for it in ntu])
    ntu_p = motion_filtered(ntu, _pair_probs(model, [it.clip for it in ntu]), cfg0)
    sessions = {"sesi15": session_replay(OLD_SESSION, cfg0), "sesi16": session_replay(SESSION16, cfg0)}
    sess_p = {name: predict(model, [rep], cfg0)[0] for name, (rep, _) in sessions.items()}
    out = {}
    for th in thresholds:
        cfg = replace(cfg0, violence_threshold=th)
        row = {}
        for name, rep in sets.items():
            m = metrics(evaluate(rep, probs[name], cfg), cfg)
            row[name] = {k: m[k] for k in ("tp", "pos", "fp", "neg")}
        hit = alarms(ntu, ntu_p, cfg)
        row["ntu"] = {"tp": int(hit[violent].sum()), "pos": int(violent.sum()),
                      "fp": int(hit[~violent].sum()), "neg": int((~violent).sum())}
        for name, (rep, ranges) in sessions.items():
            events = session_events(rep, sess_p[name], cfg)
            s = score_ranges(ranges, events)
            row[name] = {k: s[k] for k in ("tp", "pos", "fp", "neg")}
            row[name]["kejadian"] = len(events)
        out[f"ambang{th:g}"] = row
        print(f"ambang {th:g}", json.dumps(row), flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tahap", choices=("silang", "kandidat", "terpasang"))
    parser.add_argument("--varian", nargs="+", default=["semua", "tanpa-kuncian"])
    parser.add_argument("--bobot", type=float, nargs="+", default=[1.0, 3.0, 10.0])
    parser.add_argument("--ambang", type=float, nargs="+", default=[0.8, 0.85, 0.9])
    args = parser.parse_args(argv)
    KELUARAN.mkdir(parents=True, exist_ok=True)
    if args.tahap == "silang":
        crossval(args.varian, args.bobot, args.ambang)
        return 0
    if args.tahap == "terpasang":
        report = all_gates(joblib.load(OUT / "violence-selected.joblib"), args.ambang)
        (KELUARAN / "gerbang-terpasang.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        return 0
    cfg0 = load_config_snapshot(OUT / "config-baseline.json")
    u, replay = load_session(cfg0)
    for variant in args.varian:
        for w in args.bobot:
            target = KELUARAN / f"violence-sesi17-{variant}-bobot{w:g}.joblib"
            if target.is_file():
                model = joblib.load(target)
            else:
                segs = segments(u, variant)
                model = fit_with(installed_sources(DATA, OUT, cfg0), [stack(items_from_replay(replay, segs), w)])
                joblib.dump(model, target, compress=3)
            report = {"model": str(target), "gerbang": all_gates(model, args.ambang)}
            (KELUARAN / f"gerbang-{variant}-bobot{w:g}.json").write_text(json.dumps(report, indent=1),
                                                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
