"""Eksperimen A/B/C dari proposal: apakah urutan gerak (B) dan konfirmasi (C) mengubah kinerja?

    python -m latih.eksperimen_abc latih      latih model A dan B (jatuh + kekerasan) dengan data dan resep yang sama
    python -m latih.eksperimen_abc ukur       ukur A, B, C, dan aplikasi lengkap pada data uji; tulis hasil.json

Definisi (sama dengan proposal):
- A, fitur satu frame: model hanya melihat ciri pose pada saat itu.
  Jatuh: sudut badan, panjang badan, tinggi kepala di atas pinggul, rasio kotak, kaki di bawah pinggul,
  keyakinan titik tubuh, dan kelurusan lutut pada frame terakhir.
  Kekerasan: jarak kedua tangan, kedua kaki, dan tengah badan A ke badan B pada frame terakhir.
- B, fitur urutan gerak: semua ciri penilai aplikasi (jatuh 2 detik, kekerasan 1 detik), termasuk kecepatan.
- C: model dan ambang B yang sama, tetapi alarm baru muncul setelah k dari n penilaian terakhir melewati ambang
  (jatuh 3 dari 5, kekerasan 6 dari 10, seperti aplikasi).
A dan B langsung beralarm pada satu penilaian di atas ambang. A, B, dan C dilatih dengan gradient boosting pada baris
latih yang sama (resep model terpasang); yang berbeda hanya kolom cirinya. Aturan tambahan aplikasi (kuncian,
korban keluar gambar, aturan jatuh, penyaring pasangan diam) tidak dipakai di A/B/C, kecuali penyaring pasangan diam
pada B/C karena ia bagian dari jalur urutan gerak. Baris "aplikasi" = model terpasang + semua aturan, sebagai pembanding.

Data uji = data yang tidak dipakai melatih model terpasang: CAUCAFall subjek 9-10 dan Le2i (jatuh); RLVS bagian uji,
AIRTLab bagian uji, Bus Violence (kekerasan); rekaman sesi 17 November 2026 (kekerasan + kegiatan biasa kontinu,
dipakai juga untuk alarm keliru per jam).
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from sahabat.alarm import AlarmManager
from sahabat.history import EPS, History
from sahabat.pair_features import WINDOW_FEATURE_NAMES
from sahabat.person_features import PERSON_FEATURE_NAMES, person_features
from sahabat.violence import PairScore

from .latih_kekerasan_baru import Replay  # noqa: F401 - cache AIRTLab lama dipickle dari __main__
from .violence_train import Item  # noqa: F401
from .windows import ClipPairs  # noqa: F401

DATA = Path("D:/sahabat-data")
P3 = DATA / "putaran3"
OUT = DATA / "abc"
SESSION = DATA / "webcam" / "sesi-20261117-101936"
INSTALLED = Path(__file__).resolve().parent.parent / "models"

FALL_A = tuple(PERSON_FEATURE_NAMES.index(n) for n in
               ("angle_now", "torso_len_ratio", "head_above_hip", "box_aspect_now", "ankle_below_hip", "conf_now",
                "knee_straight_now"))
VIOLENCE_A = tuple(WINDOW_FEATURE_NAMES.index(f"{n}_t9") for n in ("wrist_l", "wrist_r", "ankle_l", "ankle_r", "torso"))
REFIRE_S = 5.0          # satu orang hanya beralarm jatuh sekali per sekian detik (seperti aplikasi)
FALL_AFTER_LANDING_S = 2.0
TAKE_GRACE_S = 3.0      # alarm sampai 3 detik setelah ulangan masih dihitung untuk ulangan itu


class Columns:
    """Model yang hanya membaca sebagian kolom ciri; dipakai seperti model biasa (predict_proba)."""

    def __init__(self, model, cols):
        self.model, self.cols = model, list(cols)

    def predict_proba(self, x):
        return self.model.predict_proba(np.asarray(x)[:, self.cols])


def _cfg():
    from .putar import load_config_snapshot
    return load_config_snapshot(P3 / "config-baseline.json")


# ---------------------------------------------------------------- pelatihan --------------------------------------

def violence_rows(cfg):
    from .latih_negatif_jatuh import installed_sources
    sources = installed_sources(DATA, P3, cfg)
    return [np.concatenate(parts) for parts in zip(*sources, strict=True)]


def fall_rows(cfg):
    """Baris latih model jatuh terpasang (inti NTU + UR Fall + GMDCSA, ditambah CAUCAFall subjek 1-6, label descent20)."""
    from .fall_data import stack
    from .latih_jatuh_baru import caucafall, legacy
    core, _ = legacy(DATA, cfg, P3)
    clips = []
    for clip, landing in caucafall(P3, cfg, set(range(1, 7))):
        if clip.is_fall:
            labels = clip.labels.copy()
            start = (clip.onset + landing) / 2
            labels[(clip.trace.t >= start) & (clip.trace.t <= landing + 1.0)] = 1
            clip = replace(clip, labels=labels)
        clips.append(clip)
    extra = stack(clips, {"cauca": 20})
    return [np.concatenate((a, b)) for a, b in zip(core, extra, strict=True)]


def train() -> None:
    from .violence_train import fit
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _cfg()
    x, y, w = fall_rows(cfg)
    print("jatuh", len(y), "baris", int(y.sum()), "positif", flush=True)
    for name, cols in (("A", FALL_A), ("B", tuple(range(x.shape[1])))):
        if (OUT / f"jatuh-{name}.joblib").is_file():
            continue
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                               class_weight="balanced", random_state=0)
        model.fit(x[:, list(cols)], y, sample_weight=w)
        joblib.dump(Columns(model, cols), OUT / f"jatuh-{name}.joblib", compress=3)
        print("jatuh", name, "selesai", flush=True)
    x, y, w = violence_rows(cfg)
    print("kekerasan", len(y), "baris", int(y.sum()), "positif", flush=True)
    for name, cols in (("A", VIOLENCE_A), ("B", tuple(range(x.shape[1])))):
        joblib.dump(Columns(fit(x[:, list(cols)], y, w), cols), OUT / f"kekerasan-{name}.joblib", compress=3)
        print("kekerasan", name, "selesai", flush=True)


# ---------------------------------------------------------------- penilaian jatuh ----------------------------------

def fall_ticks(trace, cfg):
    """Ciri tiap orang seluruh badan pada tiap putaran: [(t, [track_id...])] dan matriks ciri berurutan."""
    history = History(cfg)
    feats, ticks = [], []
    for t, people in zip(map(float, trace.t), trace.people):
        history.update(t, people)
        ids = []
        for track in history.tracks.values():
            if track.mode == "full" and track.samples and t - track.samples[-1].t <= EPS:
                vector = person_features(track, cfg.fall_window_s)
                if vector is not None:
                    feats.append(vector)
                    ids.append(track.track_id)
        ticks.append((t, ids))
    return ticks, np.asarray(feats, dtype=np.float32).reshape(-1, len(PERSON_FEATURE_NAMES))


def fall_alarms(ticks, probs, threshold, k, n):
    """Waktu alarm jatuh: k dari n penilaian terakhir orang itu di atas ambang; satu alarm per orang per REFIRE_S."""
    votes, last, out, i = {}, {}, [], 0
    for t, ids in ticks:
        for track_id in ids:
            window = votes.setdefault(track_id, deque(maxlen=n))
            window.append(probs[i] >= threshold)
            i += 1
            if sum(window) >= k and t - last.get(track_id, -1e9) >= REFIRE_S:
                out.append(t)
                last[track_id] = t
                window.clear()
    return out


# ---------------------------------------------------------------- penilaian kekerasan ------------------------------

def violence_events(replay, probs, cfg, rules: bool, ack_after_s: float | None = None):
    """Alarm kekerasan seperti AlarmManager aplikasi. rules=False: tanpa aturan kuncian dan korban keluar gambar."""
    clip = replay.item.clip
    windows = {}
    for (index, a, b, _, _), value in zip(clip.rows, probs, strict=True):
        windows.setdefault(index, []).append(PairScore((a, b), float(value)))
    manager, events, red_since = AlarmManager(cfg), [], None
    for i, (t, (modes, hits, seen, exiting)) in enumerate(zip(clip.t, replay.ticks, strict=True)):
        new = manager.tick(t, [], windows.get(i, []), modes, hits if rules else (),
                           seen=seen if rules and cfg.gesture_enabled else None, exiting=exiting if rules else None)
        events += [float(e.t) for e in new]
        if ack_after_s is not None:
            if manager.active is not None:
                red_since = t if red_since is None else red_since
                if t - red_since >= ack_after_s:
                    manager.acknowledge(t)
                    red_since = None
            else:
                red_since = None
    return events


def _violence_probs(model, replays, cfg, motion_filter: bool):
    from .train_rf import _pair_probs
    from .violence_train import motion_filtered
    items = [r.item for r in replays]
    probs = _pair_probs(model, [it.clip for it in items])
    return motion_filtered(items, probs, cfg) if motion_filter else probs


# ---------------------------------------------------------------- data uji ------------------------------------------

def fall_test_clips(cfg):
    from .fall_data import video_rows
    from .latih_jatuh_baru import caucafall
    from .putar import PoseTrace
    clips = [(c, landing) for c, landing in caucafall(P3, cfg, {9, 10})]
    for row in json.loads((P3 / "le2i-selection.json").read_text(encoding="utf-8")):
        tr = PoseTrace.load(P3 / "le2i_416" / (row["key"].replace("/", "__") + ".npz"))
        onset, landing = row["falls"][0] if row["falls"] else (None, None)
        clips.append((video_rows("le2i", row["key"], 0, tr, cfg, onset, landing,
                                 "jatuh" if onset is not None else "biasa"), landing))
    return clips


def violence_test_sets(cfg):
    from .banding_pose import airtlab_replay
    from .latih_kekerasan_baru import bus
    from .latih_rlvs import items as rlvs_items
    return {"rlvs": rlvs_items(P3, "test", cfg),
            "airtlab": [r for r in airtlab_replay(DATA, cfg) if r.split == "test"],
            "bus": bus(P3, cfg)}


def session(cfg):
    """Rekaman sesi 17: replay pasangan, ciri jatuh, ulangan K/B, dan rentang kegiatan biasa untuk alarm per jam."""
    from .latih_kekerasan_baru import prepare
    from .latih_sesi_jauh import IGNORE_AFTER, IGNORE_BEFORE, REAL_GAP_ALARMS, START
    from .putar import PoseTrace
    tr = PoseTrace.load(SESSION / "jejak_416.npz")
    u = json.loads((SESSION / "ulangan.json").read_text(encoding="utf-8"))
    cache = OUT / "sesi17-replay.joblib"
    if cache.is_file():
        replay = joblib.load(cache)
    else:
        replay = prepare(SESSION.name, "webcam", False, 0, "sesi", list(zip(map(float, tr.t), tr.people)),
                         (tr.frame_h, tr.frame_w), cfg)
        joblib.dump(replay, cache, compress=3)
    takes = [(k["lo"], k["hi"] + TAKE_GRACE_S, k["label"] == "kekerasan") for k in u["takes"]]
    ignore = [(t - IGNORE_BEFORE, t + IGNORE_AFTER) for t in REAL_GAP_ALARMS]
    end = float(tr.t[-1])
    # detik "kegiatan biasa": setelah START, di luar ulangan kekerasan dan zona abaikan
    grid = np.arange(START, end, 0.1)
    normal = np.ones(len(grid), dtype=bool)
    for a, b, violent in takes:
        if violent:
            normal &= ~((grid >= a) & (grid <= b))
    for a, b in ignore:
        normal &= ~((grid >= a) & (grid <= b))
    return {"trace": tr, "replay": replay, "takes": takes, "ignore": ignore, "start": START,
            "normal_hours": float(normal.sum()) * 0.1 / 3600.0, "grid": grid, "normal": normal}


def _in_normal(s, t):
    i = int(round((t - s["start"]) / 0.1))
    return 0 <= i < len(s["normal"]) and bool(s["normal"][i])


# ---------------------------------------------------------------- pengukuran ----------------------------------------

def _prf(tp, fn, fp):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3)}


def measure() -> dict:
    from .latih_jatuh_baru import measure as app_fall_measure
    cfg = _cfg()
    fall_models = {n: joblib.load(OUT / f"jatuh-{n}.joblib") for n in "AB"}
    viol_models = {n: joblib.load(OUT / f"kekerasan-{n}.joblib") for n in "AB"}
    installed_fall = joblib.load(INSTALLED / "fall_model.joblib")
    installed_viol = joblib.load(INSTALLED / "violence_model.joblib")
    configs = {"A": ("A", 1, 1, False), "B": ("B", 1, 1, True), "C": ("B", None, None, True)}

    # ---- jatuh ----
    fall_clips = fall_test_clips(cfg)
    prepared = [(clip, landing, *fall_ticks(clip.trace, cfg)) for clip, landing in fall_clips]
    s = session(cfg)
    s_ticks, s_feats = fall_ticks(s["trace"], cfg)
    print("data jatuh siap:", len(prepared), "klip;", round(s["normal_hours"], 3), "jam kegiatan biasa", flush=True)
    fall = {}
    for cname, (mname, k, n, _) in configs.items():
        k = k or cfg.fall_confirm_k
        n = n or cfg.fall_confirm_n
        model = fall_models[mname]
        tp = fn = fp = 0
        delays, scored = [], 0
        for clip, landing, ticks, feats in prepared:
            probs = model.predict_proba(feats)[:, 1] if len(feats) else np.zeros(0)
            scored += bool(len(feats))
            alarms = fall_alarms(ticks, probs, cfg.fall_model_threshold, k, n)
            if clip.is_fall:
                good = [t for t in alarms if t >= clip.onset - EPS and t <= landing + FALL_AFTER_LANDING_S]
                tp += bool(good)
                fn += not good
                fp += bool([t for t in alarms if t < clip.onset - EPS])         # alarm sebelum kejadian
                if good:
                    delays.append(min(good) - clip.onset)
            else:
                fp += bool(alarms)
        s_probs = model.predict_proba(s_feats)[:, 1] if len(s_feats) else np.zeros(0)
        s_alarms = [t for t in fall_alarms(s_ticks, s_probs, cfg.fall_model_threshold, k, n) if _in_normal(s, t)]
        row = _prf(tp, fn, fp + len(s_alarms))
        row.update({"alarm_keliru_per_jam": round(len(s_alarms) / s["normal_hours"], 2),
                    "latensi_median": round(float(np.median(delays)), 2) if delays else None,
                    "latensi_min": round(float(min(delays)), 2) if delays else None,
                    "latensi_max": round(float(max(delays)), 2) if delays else None,
                    "klip_dinilai": scored, "klip": len(prepared), "konfirmasi": f"{k} dari {n}"})
        fall[cname] = row
        print("jatuh", cname, row, flush=True)

    # aplikasi lengkap: model terpasang + aturan jatuh, lewat penilai aplikasi
    report = app_fall_measure(fall_clips, cfg, {"aplikasi": installed_fall}, {"aplikasi": 0.5})["aplikasi"]
    tp = sum(r["delay_s"] is not None for r in report["rows"] if r["fall"])
    fn = sum(r["delay_s"] is None for r in report["rows"] if r["fall"])
    fp = sum(bool(r["events"]) for r in report["rows"] if not r["fall"]) + sum(r["early"] for r in report["rows"])
    delays = [r["delay_s"] for r in report["rows"] if r["delay_s"] is not None]
    app = _prf(tp, fn, fp)
    app.update({"latensi_median": round(float(np.median(delays)), 2) if delays else None,
                "latensi_min": round(float(min(delays)), 2) if delays else None,
                "latensi_max": round(float(max(delays)), 2) if delays else None, "konfirmasi": "3 dari 5 + aturan"})
    fall["aplikasi"] = app
    print("jatuh aplikasi", app, flush=True)

    # ---- kekerasan ----
    sets = violence_test_sets(cfg)
    print("data kekerasan siap:", {k: len(v) for k, v in sets.items()}, flush=True)
    violence = {}
    variants = {**{c: (viol_models[m], k, n, mf, False) for c, (m, k, n, mf) in configs.items()},
                "aplikasi": (installed_viol, None, None, True, True)}
    for cname, (model, k, n, motion_filter, rules) in variants.items():
        c = replace(cfg, confirm_k=k or cfg.confirm_k, confirm_n=n or cfg.confirm_n)
        tp = fn = fp = 0
        per_set = {}
        for set_name, replays in sets.items():
            probs = _violence_probs(model, replays, c, motion_filter)
            hit = [bool(violence_events(r, p, c, rules)) for r, p in zip(replays, probs, strict=True)]
            pos = [h for r, h in zip(replays, hit) if r.item.violent]
            neg = [h for r, h in zip(replays, hit) if not r.item.violent]
            per_set[set_name] = {"tp": sum(pos), "pos": len(pos), "fp": sum(neg), "neg": len(neg)}
            tp += sum(pos)
            fn += len(pos) - sum(pos)
            fp += sum(neg)
        probs = _violence_probs(model, [s["replay"]], c, motion_filter)[0]
        events = violence_events(s["replay"], probs, c, rules, ack_after_s=3.0)
        k_hit = b_hit = 0
        k_n = b_n = 0
        for a, b, violent in s["takes"]:
            if a < s["start"]:
                continue
            got = any(a <= t <= b for t in events)
            if violent:
                k_n += 1
                k_hit += got
            else:
                b_n += 1
                b_hit += got
        gap = [t for t in events if _in_normal(s, t) and not any(a <= t <= b for a, b, v in s["takes"] if not v)]
        normal_alarms = [t for t in events if _in_normal(s, t)]
        per_set["sesi17"] = {"tp": k_hit, "pos": k_n, "fp": b_hit, "neg": b_n, "sela": len(gap)}
        tp += k_hit
        fn += k_n - k_hit
        fp += b_hit + len(gap)
        row = _prf(tp, fn, fp)
        row.update({"alarm_keliru_per_jam": round(len(normal_alarms) / s["normal_hours"], 2), "per_data": per_set,
                    "konfirmasi": "6 dari 10 + aturan" if rules else f"{c.confirm_k} dari {c.confirm_n}"})
        violence[cname] = row
        print("kekerasan", cname, json.dumps(row), flush=True)

    result = {"jatuh": fall, "kekerasan": violence, "jam_kegiatan_biasa": round(s["normal_hours"], 3),
              "macro_f1": {c: round((fall[c]["f1"] + violence[c]["f1"]) / 2, 3) for c in fall},
              "ambang": {"jatuh": cfg.fall_model_threshold, "kekerasan": cfg.violence_threshold}}
    (OUT / "hasil.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("macro-F1", result["macro_f1"], flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tahap", choices=("latih", "ukur"))
    args = parser.parse_args(argv)
    if args.tahap == "latih":
        train()
    else:
        measure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
