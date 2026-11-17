"""Latih ulang model kekerasan dengan sesi webcam berlabel per ulangan, lalu ukur di semua gerbang.

    python -m latih.latih_sesi_webcam reproduksi            resep model terpasang dilatih ulang; harus sama
    python -m latih.latih_sesi_webcam silang --bobot 10     validasi silang 3 lipatan pada sesi (tiap ulangan dinilai
                                                            model yang tidak melihatnya)
    python -m latih.latih_sesi_webcam kandidat --bobot 10   latih dengan seluruh sesi, ukur semua gerbang regresi
    python -m latih.latih_sesi_webcam sapu                  bobot 1 dan 3 x ambang 0,8-0,95 pada semua ukuran

Sesi: D:/sahabat-data/webcam/<SESI>/ (jejak_416.npz + label.json per ulangan).
Gerbang regresi: RLVS uji, AIRTLab uji, Bus Violence,
UT-Interaction, NTU, dan sesi webcam 15 November. Model kandidat tidak dipasang oleh skrip ini.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np

from sahabat.alarm import AlarmManager
from sahabat.violence import PairScore

from .latih_kekerasan_baru import Replay, bus, metrics, predict, prepare  # Replay: cache lama memakai __main__
from .latih_negatif_jatuh import installed_sources
from .latih_rlvs import items as rlvs_items
from .putar import PoseTrace, load_config_snapshot
from .train_rf import _pair_probs
from .violence_train import alarms, fit, motion_filtered, ntu_items, session_items, stack

DATA = Path("D:/sahabat-data")
OUT = DATA / "putaran3"
SESSION = DATA / "webcam" / "sesi-20261116-153538"
OLD_SESSION = DATA / "webcam" / "sesi-20261115-211259"
FOLDS = 3
ACK_AFTER_S = 3.0


def session_replay(folder: Path, cfg) -> tuple[Replay, list[dict]]:
    tr = PoseTrace.load(folder / "jejak_416.npz")
    replay = prepare(folder.name, "webcam", False, 0, "sesi", list(zip(map(float, tr.t), tr.people)),
                     (tr.frame_h, tr.frame_w), cfg)
    ranges = json.loads((folder / "label.json").read_text(encoding="utf-8"))
    t0 = float(tr.t[0])
    for r in ranges:
        r["t0"], r["t1"] = t0 + r["mulai"], t0 + r["selesai"]
    return replay, ranges


def session_events(replay: Replay, probs: np.ndarray, cfg) -> list[dict]:
    """Kejadian kekerasan sepanjang sesi seperti di aplikasi: "Sudah diperiksa" ditekan 3 detik setelah alarm."""
    clip = replay.item.clip
    windows: dict[int, list[PairScore]] = {}
    for (index, a, b, _, _), value in zip(clip.rows, probs, strict=True):
        windows.setdefault(index, []).append(PairScore((a, b), float(value)))
    manager, events, red_since = AlarmManager(cfg), [], None
    for i, (t, (modes, hits, seen, exiting)) in enumerate(zip(clip.t, replay.ticks, strict=True)):
        new = manager.tick(t, [], windows.get(i, []), modes, hits, seen=seen if cfg.gesture_enabled else None,
                           exiting=exiting)
        events += [{"t": e.t, "source": e.source} for e in new]
        if manager.active is not None:
            red_since = t if red_since is None else red_since
            if t - red_since >= ACK_AFTER_S:
                manager.acknowledge(t)
                red_since = None
        else:
            red_since = None
    return events


def score_ranges(ranges: list[dict], events: list[dict], only: set[int] | None = None) -> dict:
    """Kekerasan: ada kejadian dalam rentang (+3 dtk). Biasa: tidak ada kejadian. `only`: nomor grup yang dihitung."""
    out = {"tp": 0, "pos": 0, "fp": 0, "neg": 0, "rows": []}
    for r in ranges:
        if only is not None and r["grup"] not in only:
            continue
        hit = [e for e in events if r["t0"] <= e["t"] <= r["t1"] + 3.0]
        key = ("tp", "pos") if r["kekerasan"] else ("fp", "neg")
        out[key[0]] += bool(hit)
        out[key[1]] += 1
        out["rows"].append({"grup": r["grup"], "kekerasan": r["kekerasan"], "catatan": r.get("catatan", ""),
                            "kejadian": [e["source"] for e in hit]})
    return out


def train(extra: list[tuple], cfg):
    sources = installed_sources(DATA, OUT, cfg) + extra
    x, y, w = [np.concatenate(parts) for parts in zip(*sources, strict=True)]
    print(f"{len(y)} baris latih", flush=True)
    return fit(x, y, w)


def gates(model, cfg) -> dict:
    """Semua gerbang regresi untuk satu model."""
    result = {}
    sets = {"rlvs": rlvs_items(OUT, "test", cfg),
            "airtlab": [r for r in joblib.load(OUT / "airtlab-replay.joblib") if r.split == "test"],
            "bus": bus(OUT, cfg),
            "ut": joblib.load(OUT / "ut-replay.joblib")}
    from .latih_kekerasan_baru import evaluate
    for name, replay in sets.items():
        m = metrics(evaluate(replay, predict(model, replay, cfg), cfg), cfg)
        result[name] = {k: m[k] for k in ("tp", "pos", "fp", "neg")}
        print(name, result[name], flush=True)
    ntu = ntu_items(DATA / "ntu" / "pairs_cache_d4.npz")
    hit = alarms(ntu, motion_filtered(ntu, _pair_probs(model, [it.clip for it in ntu]), cfg), cfg)
    violent = np.array([it.violent for it in ntu])
    result["ntu"] = {"tp": int(hit[violent].sum()), "pos": int(violent.sum()),
                     "fp": int(hit[~violent].sum()), "neg": int((~violent).sum())}
    print("ntu", result["ntu"], flush=True)
    old, old_ranges = session_replay(OLD_SESSION, cfg)
    s = score_ranges(old_ranges, session_events(old, predict(model, [old], cfg)[0], cfg))
    result["sesi15"] = {k: s[k] for k in ("tp", "pos", "fp", "neg")}
    print("sesi15", result["sesi15"], flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tahap", choices=("reproduksi", "silang", "kandidat", "sapu"))
    parser.add_argument("--bobot", type=float, default=10.0)
    parser.add_argument("--keluaran", type=Path, default=OUT / "sesi16")
    args = parser.parse_args(argv)
    cfg = load_config_snapshot(OUT / "config-baseline.json")
    args.keluaran.mkdir(parents=True, exist_ok=True)
    installed = joblib.load(OUT / "violence-selected.joblib")
    if args.tahap == "sapu":
        sweep(keluaran=args.keluaran)
        return 0

    if args.tahap == "reproduksi":
        model = train([], cfg)
        probe = [r.item.clip for r in joblib.load(OUT / "ut-replay.joblib")[:40]]
        a = np.concatenate(_pair_probs(model, probe))
        b = np.concatenate(_pair_probs(installed, probe))
        print("selisih peluang terbesar dengan model terpasang:", float(np.abs(a - b).max()), flush=True)
        return 0

    replay, ranges = session_replay(SESSION, cfg)
    tr = PoseTrace.load(SESSION / "jejak_416.npz")
    labelled = session_items(tr, ranges, cfg)

    if args.tahap == "silang":
        report = {"bobot": args.bobot, "lipatan": []}
        base = session_events(replay, predict(installed, [replay], cfg)[0], cfg)
        report["terpasang"] = score_ranges(ranges, base)
        merged = {"tp": 0, "pos": 0, "fp": 0, "neg": 0, "rows": []}
        for fold in range(FOLDS):
            test = {r["grup"] for k, r in enumerate(ranges) if k % FOLDS == fold}
            train_items = [it for it, r in zip(labelled, ranges, strict=True) if r["grup"] not in test]
            model = train([stack(train_items, args.bobot)], cfg)
            s = score_ranges(ranges, session_events(replay, predict(model, [replay], cfg)[0], cfg), test)
            for k in ("tp", "pos", "fp", "neg"):
                merged[k] += s[k]
            merged["rows"] += s["rows"]
            print(f"lipatan {fold + 1}", {k: s[k] for k in ("tp", "pos", "fp", "neg")}, flush=True)
        report["silang"] = merged
        print("terpasang", {k: report["terpasang"][k] for k in ("tp", "pos", "fp", "neg")}, flush=True)
        print("silang   ", {k: merged[k] for k in ("tp", "pos", "fp", "neg")}, flush=True)
        (args.keluaran / f"silang-bobot{args.bobot:g}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                                                        encoding="utf-8")
        return 0

    target = args.keluaran / f"violence-sesi16-bobot{args.bobot:g}.joblib"
    if target.is_file():
        model = joblib.load(target)
    else:
        model = train([stack(labelled, args.bobot)], cfg)
        joblib.dump(model, target, compress=3)
    report = {"bobot": args.bobot, "model": str(target), "kandidat": gates(model, cfg)}
    baseline_file = args.keluaran / "gerbang-terpasang.json"
    if baseline_file.is_file():
        report["terpasang"] = json.loads(baseline_file.read_text(encoding="utf-8"))
    else:
        report["terpasang"] = gates(installed, cfg)
        baseline_file.write_text(json.dumps(report["terpasang"], indent=1), encoding="utf-8")
    (args.keluaran / f"gerbang-bobot{args.bobot:g}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0



def sweep(weights=(1.0, 3.0), thresholds=(0.8, 0.85, 0.9, 0.95), keluaran: Path = OUT / "sesi16") -> dict:
    """Bobot x ambang: validasi silang sesi dan semua gerbang regresi, peluang dihitung sekali per model."""
    from .latih_kekerasan_baru import evaluate
    cfg0 = load_config_snapshot(OUT / "config-baseline.json")
    replay, ranges = session_replay(SESSION, cfg0)
    tr = PoseTrace.load(SESSION / "jejak_416.npz")
    labelled = session_items(tr, ranges, cfg0)
    sets = {"rlvs": rlvs_items(OUT, "test", cfg0),
            "airtlab": [r for r in joblib.load(OUT / "airtlab-replay.joblib") if r.split == "test"],
            "bus": bus(OUT, cfg0), "ut": joblib.load(OUT / "ut-replay.joblib")}
    ntu = ntu_items(DATA / "ntu" / "pairs_cache_d4.npz")
    violent = np.array([it.violent for it in ntu])
    old, old_ranges = session_replay(OLD_SESSION, cfg0)
    report = {}
    for w in weights:
        folds = []
        for fold in range(FOLDS):
            test = {r["grup"] for k, r in enumerate(ranges) if k % FOLDS == fold}
            model = train([stack([it for it, r in zip(labelled, ranges, strict=True) if r["grup"] not in test], w)],
                          cfg0)
            folds.append((test, predict(model, [replay], cfg0)[0]))
        full = train([stack(labelled, w)], cfg0)
        joblib.dump(full, keluaran / f"violence-sesi16-bobot{w:g}.joblib", compress=3)
        probs = {name: predict(full, rep, cfg0) for name, rep in sets.items()}
        ntu_p = motion_filtered(ntu, _pair_probs(full, [it.clip for it in ntu]), cfg0)
        old_p = predict(full, [old], cfg0)[0]
        for th in thresholds:
            cfg = replace(cfg0, violence_threshold=th)
            row = {}
            merged = {"tp": 0, "pos": 0, "fp": 0, "neg": 0}
            for test, p in folds:
                s = score_ranges(ranges, session_events(replay, p, cfg), test)
                for k in merged:
                    merged[k] += s[k]
            row["sesi16_silang"] = merged
            for name, rep in sets.items():
                m = metrics(evaluate(rep, probs[name], cfg), cfg)
                row[name] = {k: m[k] for k in ("tp", "pos", "fp", "neg")}
            hit = alarms(ntu, ntu_p, cfg)
            row["ntu"] = {"tp": int(hit[violent].sum()), "fp": int(hit[~violent].sum())}
            s = score_ranges(old_ranges, session_events(old, old_p, cfg))
            row["sesi15"] = {k: s[k] for k in ("tp", "pos", "fp", "neg")}
            report[f"bobot{w:g}-ambang{th:g}"] = row
            print(f"bobot {w:g} ambang {th:g}", json.dumps(row), flush=True)
            (keluaran / "sapu.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
