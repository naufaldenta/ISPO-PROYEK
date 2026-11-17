"""Bandingkan model pose (YOLO11n sebagai kontrol, YOLO26n) pada seluruh dataset: ekstrak ulang jejak, latih ulang, ukur.

    python -m latih.banding_pose siapkan --pose yolo26n-pose     buat folder cermin D:/sahabat-pose/<pose>
    python -m latih.banding_pose cek                             ekstrak ulang beberapa klip dengan YOLO11n dan
                                                                 bandingkan dengan jejak lama (harus sama)
    python -m latih.banding_pose ekstrak --pose yolo26n-pose     jejak semua dataset (bisa dilanjutkan bila terhenti)
    python -m latih.banding_pose latih --pose yolo26n-pose       model kekerasan + jatuh dengan resep terpasang
    python -m latih.banding_pose ukur --pose yolo26n-pose        semua gerbang: model terpasang dan model latih ulang

Folder cermin meniru tata letak D:/sahabat-data: berkas sumber (video, label) ditautkan, folder jejak diisi baru.
Kontrol `yolo11n-pose` menautkan jejak lama apa adanya, jadi resep latih ulangnya bisa dibandingkan dengan model terpasang.

Batas yang diketahui:
- Sesi webcam 15 November hanya tersimpan sebagai jejak YOLO11n (videonya tidak ada). Resep latih tetap memakainya
  apa adanya untuk semua model pose; gerbang sesi 15 hanya berarti untuk kontrol.
- Resep kekerasan memilih jendela puncak UT dan sesi 15 dengan peluang putaran pertama. Peluang aslinya (berkas OOF)
  terikat pada jejak YOLO11n, jadi di sini diganti peluang model terpasang untuk semua model pose, termasuk kontrol.
- NTU memakai titik tubuh 2D OpenMMLab, bukan YOLO, sehingga tidak diekstrak ulang.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np

SRC = Path("D:/sahabat-data")
MIRRORS = Path("D:/sahabat-pose")
P3 = "putaran3"
SESSIONS = ("sesi-20261115-211259", "sesi-20261116-153538", "sesi-20261117-101936")
RETRACE_SESSIONS = SESSIONS[1:]               # sesi 15 tidak punya video
INSTALLED_VIOLENCE = SRC / P3 / "violence-selected.joblib"
INSTALLED_FALL = SRC / P3 / "fall-selected.joblib"
FALL_GATE = 0.5
SMALL_FILES = ("config-baseline.json", "rlvs-split.json", "rlvs-selection.json", "le2i-selection.json",
               "caucafall-labels.csv", "le2i-labels.csv", "violence-fall-negative-labels.json")
TRACE_DIRS = {  # relatif terhadap akar data: folder jejak YOLO
    "airtlab": "airtlab_pose", "gmdcsa": "gmdcsa24/jejak_416", "urfall": "urfall/jejak_416",
    "ut": "utinteraction/ekstrak/jejak_416", "bus": f"{P3}/bus_416", "cauca": f"{P3}/cauca_416",
    "le2i": f"{P3}/le2i_416", "rlvs": f"{P3}/rlvs_416"}


def _junction(link: Path, target: Path) -> None:
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


def _mirror(src: Path, dst: Path, skip: set[str]) -> None:
    """Folder nyata `dst`: berkas di-hardlink, subfolder di-junction, kecuali nama di `skip`."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in skip:
            continue
        target = dst / item.name
        if item.is_dir():
            _junction(target, item)
        elif not target.exists():
            os.link(item, target)


def prepare_mirror(pose: str) -> Path:
    root = MIRRORS / pose
    control = pose == "yolo11n-pose"
    _junction(root / "ntu", SRC / "ntu")
    _junction(root / "airtlab", SRC / "airtlab")
    _mirror(SRC / "urfall", root / "urfall", {"jejak_416"})
    _mirror(SRC / "gmdcsa24", root / "gmdcsa24", {"jejak_416"})
    _mirror(SRC / "utinteraction", root / "utinteraction", {"ekstrak"})
    _mirror(SRC / "utinteraction" / "ekstrak", root / "utinteraction" / "ekstrak", {"jejak_416"})
    out = root / P3
    out.mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES:
        if not (out / name).exists():
            shutil.copy2(SRC / P3 / name, out / name)
    if not (out / "bus-violence.zip").exists():
        os.link(SRC / P3 / "bus-violence.zip", out / "bus-violence.zip")
    for session in SESSIONS:
        folder = root / "webcam" / session
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("label.json", "ulangan.json"):
            if (SRC / "webcam" / session / name).exists() and not (folder / name).exists():
                shutil.copy2(SRC / "webcam" / session / name, folder / name)
        if (control or session not in RETRACE_SESSIONS) and not (folder / "jejak_416.npz").exists():
            os.link(SRC / "webcam" / session / "jejak_416.npz", folder / "jejak_416.npz")
    if control:
        for rel in TRACE_DIRS.values():
            _junction(root / rel, SRC / rel)
    return root


# ---- ekstraksi -------------------------------------------------------------------------

def use_pose(pose: str) -> None:
    import sahabat.pose
    sahabat.pose.MODEL_NAME = pose
    sahabat.pose.WEIGHTS = sahabat.pose.MODELS_DIR / f"{pose}.pt"


def _save(tr, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part.npz")
    tr.save(temporary)
    temporary.replace(target)


def _batched(root: Path, pose: str):
    from .jejak_dataset import batch_model
    folder = root / P3 / "batch"
    folder.mkdir(parents=True, exist_ok=True)
    return batch_model(folder, 8)


def jobs(root: Path):
    """(nama dataset, berkas tujuan, fungsi pembuat jejak) untuk semua jejak YOLO."""
    from zipfile import ZipFile
    from sahabat.config import Config
    rate = Config().rate_hz
    out = root / P3
    air = SRC / "airtlab" / "violence-detection-dataset"
    for video in sorted(p for p in air.rglob("*.mp4")):
        yield "airtlab", root / "airtlab_pose" / video.relative_to(air).with_suffix(".npz"), ("airtlab", video)
    with ZipFile(SRC / P3 / "bus-violence.zip") as archive:
        names = archive.read("test.txt").decode().splitlines()
    for name in names:
        folder = "Violence" if name.startswith("VIOLENCE_") else "NoViolence"
        yield "bus", out / "bus_416" / f"{Path(name).stem}.npz", ("batch", SRC / P3 / "bus" / folder / name, rate)
    for video in sorted((SRC / P3 / "cauca").rglob("*.avi")):
        yield "cauca", out / "cauca_416" / f"{video.stem}.npz", ("batch", video, rate)
    for row in json.loads((SRC / P3 / "le2i-selection.json").read_text(encoding="utf-8")):
        video = SRC / P3 / "le2i" / row["scene"] / (row["name"] + row["ext"])
        yield "le2i", out / "le2i_416" / (row["key"].replace("/", "__") + ".npz"), ("batch", video, 10)
    for row in json.loads((SRC / P3 / "rlvs-selection.json").read_text(encoding="utf-8")):
        video = SRC / P3 / "rlvs" / Path(row["member"]).name
        yield "rlvs", out / "rlvs_416" / f"{video.stem}.npz", ("batch", video, 10)
    for session in RETRACE_SESSIONS:
        yield "webcam", root / "webcam" / session / "jejak_416.npz", ("session", SRC / "webcam" / session)
    # Dataset dengan pembuat jejak sendiri (fase frame): dijalankan lewat fungsi aslinya pada folder cermin.
    yield "urfall", None, ("build", "urfall")
    yield "gmdcsa", None, ("build", "gmdcsa")
    yield "ut", None, ("build", "ut")


def run_job(root: Path, spec, target: Path | None, single, batch):
    from sahabat.config import Config
    from .extract_pose import extract
    from .jejak_dataset import batch_trace
    from .putar import trace
    from sahabat.recorder import read_session
    kind = spec[0]
    if kind == "airtlab":                      # AIRTLab dulu diekstrak di resolusi 480 (extract_pose bawaan)
        target.parent.mkdir(parents=True, exist_ok=True)
        extract(spec[1], target, single(480))
    elif kind == "batch":
        tr = batch_trace(spec[1], spec[2], batch(), 8)
        if not len(tr.t):
            raise ValueError(spec[1])
        _save(tr, target)
    elif kind == "session":
        pose = single()
        pose.reset()
        _save(trace(read_session(spec[1]), pose), target)
    elif kind == "build":
        rate = Config().rate_hz
        if spec[1] == "urfall":
            from .urfall import build_traces
            build_traces(root / "urfall", 416, rate)
        elif spec[1] == "gmdcsa":
            from .gmdcsa import build_traces
            build_traces(root / "gmdcsa24" / "ekstrak", 416, rate)
        else:
            from .utinteraction import build_traces
            build_traces(root / "utinteraction" / "ekstrak", 416, rate)


def extract_all(pose: str, only: set[str] | None = None) -> None:
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    use_pose(pose)
    root = prepare_mirror(pose)
    cache: dict = {}

    def single(size=416):
        if size not in cache:
            from sahabat.pose import YoloPose
            cache[size] = YoloPose(size)
        return cache[size]

    def batch():
        if "batch" not in cache:
            cache["batch"] = _batched(root, pose)
        return cache["batch"]

    todo = [j for j in jobs(root) if only is None or j[0] in only]
    started, done = time.monotonic(), 0
    for n, (name, target, spec) in enumerate(todo, 1):
        if target is not None and target.exists():
            continue
        run_job(root, spec, target, single, batch)
        done += 1
        if done % 20 == 0 or spec[0] in ("session", "build"):
            print(f"{pose} {n}/{len(todo)} {name} ({time.monotonic() - started:.0f} dtk)", flush=True)
    print(f"{pose} ekstraksi selesai ({time.monotonic() - started:.0f} dtk)", flush=True)


def check_reproduction() -> None:
    """Ekstrak ulang beberapa klip dengan YOLO11n ke folder sementara dan bandingkan dengan jejak lama."""
    use_pose("yolo11n-pose")
    scratch = MIRRORS / "_cek"
    if scratch.exists():
        shutil.rmtree(scratch)
    picked, seen = [], {}
    for name, target, spec in jobs(MIRRORS / "_jalur"):
        if target is None or name == "webcam":
            continue
        seen[name] = seen.get(name, 0) + 1
        if seen[name] <= 2:
            picked.append((name, target, spec))
    from sahabat.pose import YoloPose
    cache = {416: YoloPose(416), 480: YoloPose(480)}
    from .jejak_dataset import batch_model
    (scratch / "b").mkdir(parents=True, exist_ok=True)
    cache["batch"] = batch_model(scratch / "b", 8)
    from .putar import PoseTrace
    from .windows import load_clip
    for name, target, spec in picked:
        rel = target.relative_to(MIRRORS / "_jalur")
        mine = scratch / rel
        run_job(scratch, spec, mine, lambda size=416: cache[size], lambda: cache["batch"])
        if name == "airtlab":
            a, b = load_clip(SRC / rel), load_clip(mine)
            ta, tb = [t for t, _ in a], [t for t, _ in b]
            pa = [p.kp for _, ps in a for p in ps]
            pb = [p.kp for _, ps in b for p in ps]
        else:
            a, b = PoseTrace.load(SRC / rel), PoseTrace.load(mine)
            ta, tb = list(a.t), list(b.t)
            pa = [p.kp for ps in a.people for p in ps]
            pb = [p.kp for ps in b.people for p in ps]
        same_t = len(ta) == len(tb) and np.allclose(ta, tb)
        diff = (float(np.max(np.abs(np.array(pa) - np.array(pb)))) if len(pa) == len(pb) and pa else None)
        print(f"{name:8s} {rel.name:40s} waktu sama {same_t} orang {len(pa)}/{len(pb)} selisih titik maks {diff}",
              flush=True)
    # dataset dengan fase: bandingkan satu klip
    from .putar import trace, video_ticks, zip_ticks
    from .urfall import FPS
    pose = cache[416]
    pose.reset()
    tr = trace(zip_ticks(SRC / "urfall" / "adl-01-cam0-rgb.zip", FPS, 10.0, 1), pose)
    old = PoseTrace.load(SRC / "urfall" / "jejak_416" / "adl-01.p1.npz")
    print("urfall adl-01.p1 waktu sama", np.allclose(tr.t, old.t),
          "selisih", float(np.max(np.abs(np.array([p.kp for ps in tr.people for p in ps])
                                          - np.array([p.kp for ps in old.people for p in ps])))))


# ---- latih dan ukur --------------------------------------------------------------------

def _cfg(root: Path):
    from .putar import load_config_snapshot
    return load_config_snapshot(root / P3 / "config-baseline.json")


def violence_core(root: Path, cfg):
    """Seperti latih_kekerasan_baru.core, tetapi peluang putaran pertama UT/sesi 15 dari model terpasang."""
    from .train_rf import _pair_probs
    from .violence_train import ntu_items, stack, ut_items, webcam_items
    cache = root / P3 / "violence-core.joblib"
    if cache.is_file():
        return joblib.load(cache)
    items = ntu_items(root / "ntu" / "pairs_cache_d4.npz")
    n_ntu = len(items)
    with np.load(root / "ntu" / "oof_kekerasan.npz", allow_pickle=False) as z:
        offsets = z["offsets"]
        first = np.split(z["first"], offsets[1:-1])[:n_ntu]
    if [len(f) for f in first] != [len(it.clip.rows) for it in items]:
        raise ValueError("Cache peluang NTU tidak cocok dengan baris NTU")
    extra = ut_items(root / "utinteraction" / "ekstrak", cfg)
    extra += webcam_items(root / "webcam" / SESSIONS[0], cfg)
    installed = joblib.load(INSTALLED_VIOLENCE)
    first += [np.asarray(p) for p in _pair_probs(installed, [it.clip for it in extra])]
    rows = stack(items + extra, 10, first)
    joblib.dump(rows, cache, compress=3)
    print("core kekerasan", len(rows[1]), flush=True)
    return rows


def airtlab_replay(root: Path, cfg):
    from .latih_kekerasan_baru import airtlab
    cache = root / P3 / "airtlab-replay.joblib"
    if cache.is_file():
        return joblib.load(cache)
    replay = airtlab(root, cfg)
    joblib.dump(replay, cache, compress=3)
    return replay


def ut_replay(root: Path, cfg):
    from .latih_kekerasan_baru import prepare
    from .utinteraction import build_traces, load_index
    cache = root / P3 / "ut-replay.joblib"
    if cache.is_file():
        return joblib.load(cache)
    folder = root / "utinteraction" / "ekstrak"
    index = {c.name: c for c in load_index(folder)}
    clips = []
    for (name, phase), tr in sorted(build_traces(folder, 416, cfg.rate_hz).items()):
        label = index[name]
        clips.append(prepare(f"{name}.p{phase}", "ut", label.violent, 1000 + 100 * label.set_no + label.sequence,
                             "regression", list(zip(map(float, tr.t), tr.people)), (tr.frame_h, tr.frame_w), cfg))
    joblib.dump(clips, cache, compress=3)
    return clips


def train_models(pose: str) -> None:
    from sklearn.base import clone  # noqa: F401
    from .latih_jatuh_baru import caucafall, fit_candidate, legacy
    from .latih_negatif_jatuh import installed_sources
    from .violence_train import fit
    root = MIRRORS / pose
    cfg = _cfg(root)
    target = root / P3 / "violence-latih-ulang.joblib"
    if not target.is_file():
        violence_core(root, cfg)          # tersimpan di tempat yang dibaca core() oleh installed_sources
        airtlab_replay(root, cfg)
        sources = installed_sources(root, root / P3, cfg)
        x, y, w = [np.concatenate(parts) for parts in zip(*sources, strict=True)]
        print(pose, "kekerasan", len(y), "baris", flush=True)
        joblib.dump(fit(x, y, w), target, compress=3)
    target = root / P3 / "fall-latih-ulang.joblib"
    if not target.is_file():
        core, _videos = legacy(root, cfg, root / P3)
        model = fit_candidate(core, list(caucafall(root / P3, cfg, set(range(1, 7)))), "descent20")
        model.fall_rule_support_threshold_ = FALL_GATE
        joblib.dump(model, target, compress=3)
    print(pose, "latih selesai", flush=True)


def violence_gates(root: Path, models: dict, thresholds=(0.8,)) -> dict:
    from .latih_kekerasan_baru import bus, evaluate, metrics, predict
    from .latih_rlvs import items as rlvs_items
    from .latih_sesi_jauh import scoring_ranges
    from .latih_sesi_webcam import score_ranges, session_events, session_replay
    from .latih_kekerasan_baru import prepare
    from .putar import PoseTrace
    from .train_rf import _pair_probs
    from .violence_train import alarms, motion_filtered, ntu_items
    cfg0 = _cfg(root)
    sets = {"rlvs": rlvs_items(root / P3, "test", cfg0),
            "airtlab": [r for r in airtlab_replay(root, cfg0) if r.split == "test"],
            "bus": bus(root / P3, cfg0), "ut": ut_replay(root, cfg0)}
    ntu = ntu_items(root / "ntu" / "pairs_cache_d4.npz")
    violent = np.array([it.violent for it in ntu])
    sessions = {}
    for session in SESSIONS[:2]:
        sessions[session[5:13]] = session_replay(root / "webcam" / session, cfg0)
    folder = root / "webcam" / SESSIONS[2]
    tr = PoseTrace.load(folder / "jejak_416.npz")
    u = json.loads((folder / "ulangan.json").read_text(encoding="utf-8"))
    sessions["20261117"] = (prepare(folder.name, "webcam", False, 0, "sesi", list(zip(map(float, tr.t), tr.people)),
                                    (tr.frame_h, tr.frame_w), cfg0), scoring_ranges(u))
    result = {}
    for name, model in models.items():
        probs = {k: predict(model, rep, cfg0) for k, rep in sets.items()}
        ntu_p = motion_filtered(ntu, _pair_probs(model, [it.clip for it in ntu]), cfg0)
        sess_p = {k: predict(model, [rep], cfg0)[0] for k, (rep, _) in sessions.items()}
        for th in thresholds:
            cfg = replace(cfg0, violence_threshold=th)
            row = {}
            for k, rep in sets.items():
                m = metrics(evaluate(rep, probs[k], cfg), cfg)
                row[k] = {key: m[key] for key in ("tp", "pos", "fp", "neg")}
            hit = alarms(ntu, ntu_p, cfg)
            row["ntu"] = {"tp": int(hit[violent].sum()), "pos": int(violent.sum()),
                          "fp": int(hit[~violent].sum()), "neg": int((~violent).sum())}
            for k, (rep, ranges) in sessions.items():
                events = session_events(rep, sess_p[k], cfg)
                s = score_ranges(ranges, events)
                row[f"sesi{k[-2:]}"] = {key: s[key] for key in ("tp", "pos", "fp", "neg")} | {"kejadian": len(events)}
            result[f"{name}@{th:g}"] = row
            print(root.name, f"{name}@{th:g}", json.dumps(row), flush=True)
    return result


def fall_gates(root: Path, models: dict) -> dict:
    from .fall_data import video_rows
    from .latih_jatuh_baru import caucafall, legacy, measure
    from .putar import PoseTrace
    cfg = _cfg(root)
    _core, videos = legacy(root, cfg, root / P3)
    clips = list(caucafall(root / P3, cfg, {9, 10})) + [(clip, None) for clip in videos]
    for row in json.loads((root / P3 / "le2i-selection.json").read_text(encoding="utf-8")):
        tr = PoseTrace.load(root / P3 / "le2i_416" / (row["key"].replace("/", "__") + ".npz"))
        onset, landing = row["falls"][0] if row["falls"] else (None, None)
        clips.append((video_rows("le2i", row["key"], 0, tr, cfg, onset, landing,
                                 "jatuh" if onset is not None else "biasa"), landing))
    report = measure(clips, cfg, models, {name: FALL_GATE for name in models})
    return {name: r["metrics"] for name, r in report.items()}


def measure_all(pose: str) -> None:
    root = MIRRORS / pose
    violence = {"terpasang": joblib.load(INSTALLED_VIOLENCE)}
    fall = {"terpasang": joblib.load(INSTALLED_FALL)}
    if (root / P3 / "violence-latih-ulang.joblib").is_file():
        violence["latih-ulang"] = joblib.load(root / P3 / "violence-latih-ulang.joblib")
        fall["latih-ulang"] = joblib.load(root / P3 / "fall-latih-ulang.joblib")
    report = {"pose": pose, "kekerasan": violence_gates(root, violence, (0.8, 0.85, 0.9)),
              "jatuh": fall_gates(root, fall)}
    (root / "gerbang.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tahap", choices=("siapkan", "cek", "ekstrak", "latih", "ukur"))
    parser.add_argument("--pose", default="yolo26n-pose")
    parser.add_argument("--hanya", nargs="+")
    args = parser.parse_args(argv)
    if args.tahap == "siapkan":
        print(prepare_mirror(args.pose))
    elif args.tahap == "cek":
        prepare_mirror("_jalur")
        check_reproduction()
    elif args.tahap == "ekstrak":
        extract_all(args.pose, set(args.hanya) if args.hanya else None)
    elif args.tahap == "latih":
        prepare_mirror(args.pose)
        train_models(args.pose)
    else:
        prepare_mirror(args.pose)
        measure_all(args.pose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
