"""Melatih penilai jatuh dari kerangka NTU RGB+D ditambah video sungguhan yang dilewatkan YOLO-pose.

    python -m latih.fall_train <ntu120_2d.pkl> [--urfall <folder>] [--gmdcsa <folder ekstrak>] [--pekerja 4]

NTU (kerangka HRNet, kamera Kinect) memberi jumlah: 946 klip jatuh dan ribuan gerakan mirip. Video sungguhan
(UR Fall Detection, GMDCSA-24) memberi kemiripan dengan aplikasi: titik tubuhnya dari YOLO-pose + ByteTrack
yang sama, dengan getar dan titik hilang yang sama. Baris dari video diberi bobot VIDEO_WEIGHT karena
jumlahnya jauh lebih sedikit.

Positif: sekitar saat orangnya rebah (0,2 dtk sebelum sampai 1 dtk sesudah). Negatif: semua langkah klip
bukan-jatuh (duduk, jongkok, membungkuk, berbaring dengan sengaja, ...) dan bagian klip jatuh sebelum jatuhnya.

Klip NTU dan UR Fall berakhir tak lama setelah orangnya rebah, jadi pose terakhirnya ditahan 1,5 detik.

Yang dilaporkan: berapa klip yang memunculkan ALARM, dari validasi silang (tiap klip dinilai model yang tidak
pernah melihat pemeran/subjek/klip itu), dan dari uji lintas-set (model tidak pernah melihat set videonya).
"""
from __future__ import annotations

import sys
import tempfile
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sahabat.config import MODELS_DIR, Config

from .fall_data import ClipRows, clip_features, fold_of, landing_time, stack, video_rows
from .ntu import FALL, NOT_FALL, parse_name, safe_load, to_frames
from .ntu_fall import fall_fired
from .putar import replay

__all__ = ["clip_features", "confirmed", "tick_labels", "wanted"]

MODEL_PATH = MODELS_DIR / "fall_model.joblib"
REPORT_PATH = MODELS_DIR / "fall_model.laporan.txt"
HOLD_S = 1.5
SINGLE_PERSON = set(range(1, 50)) | set(range(61, 106))        # aksi satu-orang di NTU RGB+D 120
OTHER_KEEP_1_IN = 10                                           # cuplikan aksi lain: 1 dari 10 klip
POSITIVE_FROM = -2                                             # langkah positif: dari 2 langkah sebelum akhir klip...
POSITIVE_TO = 10                                               # ...sampai 1 detik sesudahnya
N_FOLDS = 5
VIDEO_WEIGHT = 10.0                                            # bobot baris video sungguhan terhadap baris NTU
IMGSZ = 416
VIDEO_HOLD_S = {"urfall": HOLD_S, "gmdcsa": 0.0}
THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95)
CONFIRMS = ((2, 3), (3, 5), (5, 8))


def wanted(frame_dir: str, action: int) -> bool:
    if action in FALL or action in NOT_FALL:
        return True
    return action in SINGLE_PERSON and zlib.crc32(frame_dir.encode()) % OTHER_KEEP_1_IN == 0


def tick_labels(n_real: int, n_total: int, is_fall: bool) -> np.ndarray:
    """1 = jatuh, 0 = bukan, -1 = tidak dipakai melatih (bagian klip jatuh sebelum orangnya rebah)."""
    if not is_fall:
        return np.zeros(n_total, dtype=np.int8)
    labels = np.full(n_total, -1, dtype=np.int8)
    labels[max(0, n_real + POSITIVE_FROM):min(n_total, n_real + POSITIVE_TO + 1)] = 1
    return labels


def confirmed(flags: np.ndarray, k: int, n: int) -> bool:
    """Apakah ada n langkah berurutan yang memuat minimal k tanda positif."""
    if len(flags) == 0:
        return False
    window = np.convolve(flags.astype(int), np.ones(n, dtype=int))
    return bool(window.max() >= k)


# ---- sumber data ------------------------------------------------------------------------

def _ntu_work(item):
    keypoint, score, frame_h = item
    cfg = Config()
    real = to_frames(keypoint, score)
    frames = to_frames(keypoint, score, hold_last_s=HOLD_S)
    return clip_features(frames, cfg), len(real), fall_fired(frames, cfg, frame_h)


def ntu_clips(pkl: Path, workers: int) -> list[ClipRows]:
    """Klip satu-orang NTU. Cirinya disimpan di sebelah berkas .pkl supaya percobaan berikutnya cepat."""
    cache = Path(pkl).with_name("fall_ciri.npz")
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            feats, offsets, n_real, rule = z["feats"], z["offsets"], z["n_real"], z["rule"]
            actions, performers, names = z["actions"], z["performers"], z["names"]
    else:
        annotations = [a for a in safe_load(Path(pkl))["annotations"] if wanted(a["frame_dir"], a["label"] + 1)]
        meta = [parse_name(a["frame_dir"]) for a in annotations]
        names = np.array([a["frame_dir"] for a in annotations])
        items = [(a["keypoint"], a["keypoint_score"], a["img_shape"][0]) for a in annotations]
        del annotations
        with Pool(workers) as pool:
            results = pool.map(_ntu_work, items, chunksize=64)
        del items
        feats = np.concatenate([r[0] for r in results])
        offsets = np.cumsum([0] + [len(r[0]) for r in results])
        n_real = np.array([r[1] for r in results])
        rule = np.array([r[2] for r in results])
        actions = np.array([m["action"] for m in meta])
        performers = np.array([m["performer"] for m in meta])
        np.savez_compressed(cache, feats=feats, offsets=offsets, n_real=n_real, rule=rule, actions=actions,
                            performers=performers, names=names)
    clips = []
    for i in range(len(n_real)):
        f = feats[offsets[i]:offsets[i + 1]]
        is_fall = int(actions[i]) in FALL
        clips.append(ClipRows("ntu", str(names[i]), int(performers[i]), is_fall, str(int(actions[i])), f,
                              tick_labels(int(n_real[i]), len(f), is_fall), bool(rule[i]), None, None))
    return clips


def urfall_clips(folder: Path, cfg: Config) -> list[ClipRows]:
    from .urfall import build_traces, fall_times, hold, lying_clips
    folder = Path(folder)
    times = fall_times(folder / "urfall-cam0-falls.csv")
    lying = lying_clips(folder / "urfall-cam0-adls.csv")
    clips = []
    for (name, phase), tr in sorted(build_traces(folder, IMGSZ, cfg.rate_hz).items()):
        number = int(name.split("-")[1])
        onset, landing = times.get(name, (None, None))
        if name.startswith("fall") and onset is None:
            continue
        kind = "jatuh" if onset is not None else ("berbaring" if name in lying else "biasa")
        group = number if name.startswith("fall") else 100 + number
        clips.append(video_rows("urfall", f"{name}.p{phase}", group, hold(tr, HOLD_S, cfg.rate_hz), cfg,
                                onset, landing, kind))
    return clips


def gmdcsa_clips(root: Path, cfg: Config) -> list[ClipRows]:
    from .gmdcsa import build_traces, load_index
    index = {c.name: c for c in load_index(root)}
    clips = []
    for (name, phase), tr in sorted(build_traces(Path(root), IMGSZ, cfg.rate_hz).items()):
        c = index[name]
        if c.is_fall and c.onset is None:                # dua klip tanpa waktu jatuh dari penyusunnya
            continue
        landing = landing_time(tr, c.onset) if c.is_fall else None
        kind = "jatuh" if c.is_fall else ("berbaring" if c.lying else "biasa")
        clips.append(video_rows("gmdcsa", f"{name}.p{phase}", c.subject, tr, cfg, c.onset, landing, kind))
    return clips


# ---- melatih dan menilai ----------------------------------------------------------------

def fit(clips: list[ClipRows], video_weight: float = VIDEO_WEIGHT):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, y, w = stack(clips, {"urfall": video_weight, "gmdcsa": video_weight})
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                          class_weight="balanced", random_state=0).fit(X, y, sample_weight=w)


def tick_probs(model, clip: ClipRows) -> np.ndarray:
    p = np.zeros(len(clip.feats))
    ok = np.isfinite(clip.feats).any(axis=1)
    if ok.any():
        p[ok] = model.predict_proba(clip.feats[ok])[:, 1]
    return p


def video_alarm(model_path: Path | None, clip: ClipRows, cfg: Config) -> bool:
    """Apakah klip video ini memunculkan alarm jatuh yang benar/keliru, lewat mesin yang sama dengan aplikasi.

    Klip jatuh: hanya alarm sesudah awal jatuh yang dihitung. Klip lain: alarm jatuh kapan pun.
    """
    alarms = [a.t for a in replay(cfg, clip.trace, fall_model=model_path, ack_after_s=None) if a.kind == "fall"]
    return any(t >= clip.onset for t in alarms) if clip.is_fall else bool(alarms)


def cross_validate(clips: list[ClipRows], cfg: Config, video_weight: float, train_on=("ntu", "urfall", "gmdcsa"),
                   log=print) -> tuple[list[np.ndarray], dict[str, bool]]:
    """(peluang per langkah untuk tiap klip NTU, {nama klip video: beralarm}) dari model yang tak melihat klipnya."""
    import joblib
    probs: list[np.ndarray] = [np.zeros(0)] * len(clips)
    alarm: dict[str, bool] = {}
    folds = np.array([fold_of(c, N_FOLDS) for c in clips])
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(N_FOLDS):
            started = time.monotonic()
            model = fit([c for c, f in zip(clips, folds) if f != k and c.source in train_on], video_weight)
            path = Path(tmp) / f"lipatan{k}.joblib"
            joblib.dump(model, path)
            for i in np.flatnonzero(folds == k):
                if clips[i].trace is None:
                    probs[i] = tick_probs(model, clips[i])
                else:
                    alarm[clips[i].name] = video_alarm(path, clips[i], cfg)
            log(f"  lipatan {k + 1}/{N_FOLDS} selesai ({time.monotonic() - started:.0f} dtk)")
    return probs, alarm


def _pct(hit, sel) -> str:
    sel = np.asarray(sel, dtype=bool)
    return f"{100.0 * np.asarray(hit)[sel].mean():5.1f}% ({int(np.asarray(hit)[sel].sum())}/{int(sel.sum())})" \
        if sel.any() else "    -"


def video_lines(clips: list[ClipRows], alarm: dict[str, bool]) -> list[str]:
    lines = []
    for source, title in (("urfall", "UR Fall Detection"), ("gmdcsa", "GMDCSA-24 (webcam laptop)")):
        mine = [c for c in clips if c.source == source and c.name in alarm]
        if not mine:
            continue
        hit = np.array([alarm[c.name] for c in mine])
        kind = np.array([c.kind for c in mine])
        lines.append(f"  {title:28s} jatuh {_pct(hit, kind == 'jatuh')}   berbaring sengaja {_pct(hit, kind == 'berbaring')}"
                     f"   kegiatan lain {_pct(hit, kind == 'biasa')}")
    return lines


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    import joblib

    args, opts = [], {}
    rest = list(argv)
    while rest:
        a = rest.pop(0)
        if a.startswith("--") and rest:
            opts[a[2:]] = rest.pop(0)
        else:
            args.append(a)
    workers = int(opts.get("pekerja", 4))
    started = time.monotonic()
    cfg = Config()

    clips = ntu_clips(Path(args[0]), workers)
    print(f"NTU: {len(clips)} klip ({time.monotonic() - started:.0f} dtk)", flush=True)
    if "urfall" in opts:
        clips += urfall_clips(Path(opts["urfall"]), cfg)
    if "gmdcsa" in opts:
        clips += gmdcsa_clips(Path(opts["gmdcsa"]), cfg)
    video = [c for c in clips if c.trace is not None]
    print(f"video sungguhan: {len(video)} putaran ({time.monotonic() - started:.0f} dtk)", flush=True)

    ntu = [c for c in clips if c.source == "ntu"]
    actions = np.array([int(c.kind) for c in ntu])
    is_fall = np.array([c.is_fall for c in ntu])
    similar = np.isin(actions, list(NOT_FALL))
    other = ~is_fall & ~similar
    rule = np.array([c.rule for c in ntu]) if cfg.fall_rule_with_model else np.zeros(len(ntu), dtype=bool)
    default = (cfg.fall_model_threshold, cfg.fall_confirm_k, cfg.fall_confirm_n)

    X, y, _ = stack(clips)
    lines = [
        f"sumber: {args[0]}" + "".join(f", {k}={v}" for k, v in opts.items() if k in ("urfall", "gmdcsa")),
        f"klip NTU: jatuh={int(is_fall.sum())}  gerakan mirip={int(similar.sum())}  aksi lain (cuplikan)={int(other.sum())}",
        f"putaran video: " + ", ".join(f"{s} {sum(c.source == s for c in video)}" for s in ("urfall", "gmdcsa")),
        f"baris latih: jatuh={int((y == 1).sum())}  bukan={int((y == 0).sum())}; bobot baris video {VIDEO_WEIGHT:g}",
        f"aturan seluruh-badan ikut memicu: {'ya' if cfg.fall_rule_with_model else 'tidak'}",
        "",
        "Video sungguhan lewat mesin aplikasi (ambang bawaan), klip yang memunculkan ALARM jatuh:",
    ]

    print("model lama (hanya NTU), semua video:", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        old = Path(tmp) / "ntu.joblib"
        joblib.dump(fit(ntu), old)
        lines += ["- dilatih hanya dari NTU (model sebelumnya):"] + \
            video_lines(video, {c.name: video_alarm(old, c, cfg) for c in video})
        for held, others in (("gmdcsa", ("ntu", "urfall")), ("urfall", ("ntu", "gmdcsa"))):
            test = [c for c in video if c.source == held]
            if test and any(c.source in others and c.trace is not None for c in clips):
                path = Path(tmp) / f"tanpa-{held}.joblib"
                joblib.dump(fit([c for c in clips if c.source in others]), path)
                lines += [f"- lintas-set: dilatih tanpa {held}, diuji pada {held}:"] + \
                    video_lines(test, {c.name: video_alarm(path, c, cfg) for c in test})
    print("\n".join(lines[-7:]), flush=True)

    print("validasi silang, semua sumber:", flush=True)
    probs, alarm = cross_validate(clips, cfg, VIDEO_WEIGHT)
    ntu_probs = [p for p, c in zip(probs, clips) if c.source == "ntu"]
    lines += [f"- validasi silang {N_FOLDS} lipatan, dilatih dari semua sumber (model yang dipasang):"] + video_lines(video, alarm)
    lines += ["", f"Klip NTU yang memunculkan ALARM jatuh (validasi silang {N_FOLDS} lipatan per pemeran, model semua sumber):",
              "  ambang  konfirmasi   jatuh       gerakan mirip   aksi lain   | dengan aturan: jatuh  mirip  lain"]
    hits = {}
    for thr in THRESHOLDS:
        for k, n in CONFIRMS:
            hit = np.array([confirmed(p >= thr, k, n) for p in ntu_probs])
            hits[(thr, k, n)] = hit
            both = hit | rule
            mark = "  <- bawaan aplikasi" if (thr, k, n) == default else ""
            lines.append(f"  {thr:<6}  {k} dari {n:<3}   {100 * hit[is_fall].mean():5.1f}%      {100 * hit[similar].mean():5.1f}%"
                         f"          {100 * hit[other].mean():5.1f}%      |      {100 * both[is_fall].mean():5.1f}%"
                         f"  {100 * both[similar].mean():5.1f}%  {100 * both[other].mean():5.1f}%{mark}")
    if default in hits:
        hit = hits[default] | rule
        lines += ["", f"Per aksi NTU pada bawaan aplikasi, ambang {default[0]}, {default[1]} dari {default[2]}:"]
        for action, name in {**FALL, **NOT_FALL}.items():
            sel = actions == action
            lines.append(f"  A{action:03d} {name:28s} {int(hit[sel].sum()):4d}/{int(sel.sum()):<4d} ({100 * hit[sel].mean():5.1f}%)")

    final = fit(clips)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH, compress=3)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nmodel disimpan: {MODEL_PATH}  ({time.monotonic() - started:.0f} dtk)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
