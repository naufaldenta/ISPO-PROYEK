"""Model kamera tinggi (pojok atau langit-langit, sekitar 2,2-3 m): data, kandidat, gerbang, dan rekaman sekolah.

    python -m latih.kamera_tinggi jejak       jejak pose Bus Violence bagian latih dan Surveillance Camera Fight
                                              (bisa dilanjutkan bila terhenti)
    python -m latih.kamera_tinggi kekerasan [--pasang]
                                              kandidat model kekerasan tinggi, gerbang, dan hasilnya; --pasang
                                              memasang kandidat terpilih hanya bila lolos gerbang
    python -m latih.kamera_tinggi jatuh [--pasang]
                                              sama untuk model jatuh (resep terpasang, bobot kamera tinggi dinaikkan)
    python -m latih.kamera_tinggi sekolah SESI... [--latih] [--pasang]
                                              rekaman sekolah (data/rekaman/<sesi>, "rekam uji" atau "rekam
                                              kegiatan biasa"): ukur model sekarang dan model tinggi; --latih
                                              melatih ulang model kekerasan tinggi dengan rekaman itu (validasi
                                              silang 3 lipatan) dan menjalankan gerbang; --pasang bila lolos

Model sekarang tidak diubah. Model tinggi adalah berkas terpisah dan hanya dipasang bila lebih baik pada data kamera
tinggi yang tidak ikut dilatih. Data:
- Bus Violence `train.txt` (1.120 klip, kamera atap bus): latih + dev kandidat kekerasan; `test.txt` tetap data uji.
- Surveillance Camera Fight (300 klip CCTV, MIT, Akti dkk. 2019): uji lintas dataset, tidak pernah dilatih.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from zipfile import ZipFile

DATA = Path("D:/sahabat-data")
P3 = DATA / "putaran3"
ROOT = DATA / "kamera-tinggi"
BUS_ZIP = P3 / "bus-violence.zip"
DEV_FRAC = 0.2
# Nomor grup yang tidak bertabrakan dengan data lama: AIRTLab 3000-4999, Bus uji 5000-5279, sesi webcam 9000-an.
BUS_TRAIN_GROUP, SCFD_GROUP = 10000, 20000
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "uji" / "kamera-tinggi"
BOBOT = (2, 5, 10)            # bobot jendela video kamera tinggi, sama dengan resep peak10 (puncak model terpasang)
BOBOT_JATUH = (40, 80)        # bobot CAUCAFall; model terpasang (descent20) memakai 20
SESI16 = DATA / "webcam" / "sesi-20261116-153538"


def _check_name(name: str) -> str:
    if Path(name).name != name or not name.endswith(".mp4"):
        raise ValueError(f"Nama video Bus Violence tidak valid: {name}")
    return name


def bus_train_videos(zip_path: Path = BUS_ZIP) -> list[tuple[str, bool]]:
    """(nama video, kekerasan?) dari `train.txt` resmi Bus Violence."""
    with ZipFile(zip_path) as archive:
        names = [n.strip() for n in archive.read("train.txt").decode().splitlines() if n.strip()]
    return [(_check_name(n), n.startswith("VIOLENCE_")) for n in names]


def ekstrak_bus_train(zip_path: Path = BUS_ZIP, out: Path = ROOT / "bus_train") -> list[Path]:
    """Video bagian latih ke out/{Violence,NoViolence}/; berkas yang sudah utuh dilewati."""
    paths = []
    with ZipFile(zip_path) as archive:
        for name, violent in bus_train_videos(zip_path):
            folder = "Violence" if violent else "NoViolence"
            source = f"{folder}/{name}"
            target = out / folder / name
            if not target.exists() or target.stat().st_size != archive.getinfo(source).file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".part")
                temporary.write_bytes(archive.read(source))
                temporary.replace(target)
            paths.append(target)
    return paths


def split_dev(names: list[str], frac: float = DEV_FRAC) -> set[str]:
    """Bagian dev yang deterministik (hash nama), tidak bergantung urutan atau versi Python."""
    return {n for n in names if int(hashlib.sha1(n.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac}


def scfd_videos(folder: Path = ROOT / "scfd") -> list[tuple[Path, bool]]:
    """(video, perkelahian?) Surveillance Camera Fight: fight/ dan noFight/."""
    out = []
    for sub, fight in (("fight", True), ("noFight", False)):
        out += [(p, fight) for p in sorted((folder / sub).glob("*.mp4"))]
    return out


def trace_target(video: Path) -> Path:
    """Berkas jejak untuk video data kamera tinggi."""
    video = Path(video)
    if video.parent.parent == ROOT / "bus_train":
        return ROOT / "bus_train_416" / f"{video.stem}.npz"
    if video.parent.parent == ROOT / "scfd":
        return ROOT / "scfd_416" / video.parent.name / f"{video.stem}.npz"
    raise ValueError(f"Bukan video data kamera tinggi: {video}")


def jejak() -> int:
    """Jejak pose CPU OpenVINO batch 8, sama dengan jejak bus_416 yang sudah ada."""
    import cv2
    import torch
    from sahabat.config import Config
    from .jejak_dataset import batch_model, batch_trace
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    videos = ekstrak_bus_train()
    scfd = [p for p, _ in scfd_videos()]
    if not scfd:
        print("Surveillance Camera Fight belum ada di", ROOT / "scfd", "(uji lintas dataset dilewati).")
    todo = videos + scfd
    model = batch_model(P3, 8)
    rate = Config().rate_hz
    started = time.monotonic()
    for i, video in enumerate(todo, 1):
        target = trace_target(video)
        if not target.is_file():
            tr = batch_trace(video, rate, model, 8)
            if not len(tr.t):
                print("Tidak ada frame:", video)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".part.npz")
            tr.save(temporary)
            temporary.replace(target)
        if i % 50 == 0 or i == len(todo):
            print(f"{i}/{len(todo)} ({time.monotonic() - started:.0f} dtk)", flush=True)
    return 0


# ---- gerbang dan pemasangan ----------------------------------------------------------

def lolos_tinggi(sekarang: dict, tinggi: dict) -> bool:
    """Model tinggi harus menangkap LEBIH BANYAK kejadian dan alarm kelirunya tidak lebih banyak, baik pada klip biasa
    data uji kamera tinggi maupun per jam pada rekaman kegiatan biasa."""
    return (tinggi["tertangkap"] > sekarang["tertangkap"] and tinggi["keliru_klip"] <= sekarang["keliru_klip"]
            and tinggi["keliru_per_jam"] <= sekarang["keliru_per_jam"])


def pasang(kandidat: Path, target: Path, lolos: bool) -> bool:
    """Salin kandidat ke models/ hanya bila lolos gerbang."""
    if not lolos:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    shutil.copyfile(kandidat, temporary)
    temporary.replace(target)
    return True


def cek_bocor(train: list, dev: list, test: list) -> None:
    """Klip uji tidak boleh ikut latih/dev, termasuk lewat grup (subjek, adegan, atau video sumber) yang sama."""
    for name_a, a, name_b, b in (("latih", train, "uji", test), ("dev", dev, "uji", test), ("latih", train, "dev", dev)):
        names = {r.item.name for r in a} & {r.item.name for r in b}
        groups = {r.item.group for r in a} & {r.item.group for r in b}
        if names or groups:
            raise ValueError(f"Data {name_a} bocor ke {name_b}: klip {sorted(names)[:3]}, grup {sorted(groups)[:3]}")


# ---- kekerasan ---------------------------------------------------------------------------

def susun_kekerasan(bus_train: list, airtlab: list, bus_test: list, scfd: list) -> tuple[list, list, list]:
    """(latih tambahan, dev, uji kamera tinggi). Latih: Bus latih (selain dev) + AIRTLab latih positif (model terpasang
    hanya memakai negatif AIRTLab). Dev: bagian dev Bus latih. Uji: Bus uji, AIRTLab uji, SCFD (lintas dataset)."""
    train = [r for r in bus_train if r.split == "train"] + \
        [r for r in airtlab if r.split == "train" and r.item.violent]
    dev = [r for r in bus_train if r.split == "dev"]
    test = list(bus_test) + [r for r in airtlab if r.split == "test"] + list(scfd)
    cek_bocor(train, dev, test)
    return train, dev, test


def _replays(cfg) -> tuple[list, list]:
    """Replay Bus latih dan SCFD (disimpan di ROOT/replay-kekerasan.joblib)."""
    import joblib
    from .latih_kekerasan_baru import prepare
    from .putar import PoseTrace
    cache = ROOT / "replay-kekerasan.joblib"
    if cache.is_file():
        return joblib.load(cache)
    names = bus_train_videos()
    dev = split_dev([n for n, _ in names])
    bus = []
    for i, (name, violent) in enumerate(names):
        path = ROOT / "bus_train_416" / f"{Path(name).stem}.npz"
        if not path.is_file():
            continue
        tr = PoseTrace.load(path)
        bus.append(prepare(name, "bus_latih", violent, BUS_TRAIN_GROUP + i, "dev" if name in dev else "train",
                           list(zip(map(float, tr.t), tr.people)), (tr.frame_h, tr.frame_w), cfg))
    scfd = []
    for i, (video, fight) in enumerate(scfd_videos()):
        path = trace_target(video)
        if not path.is_file():
            continue
        tr = PoseTrace.load(path)
        scfd.append(prepare(f"scfd/{video.parent.name}/{video.name}", "scfd", fight, SCFD_GROUP + i, "test",
                            list(zip(map(float, tr.t), tr.people)), (tr.frame_h, tr.frame_w), cfg))
    temporary = cache.with_suffix(".part")
    joblib.dump((bus, scfd), temporary, compress=3)
    temporary.replace(cache)
    return bus, scfd


def violence_sets(data: Path = DATA, cfg=None) -> tuple[list, list, list]:
    import joblib
    from . import latih_kekerasan_baru as kb
    from .latih_kekerasan_baru import Replay
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "Replay"):
        main_module.Replay = Replay         # cache AIRTLab lama di-pickle dengan Replay dari __main__
    bus_train, scfd = _replays(cfg)
    airtlab = joblib.load(data / "putaran3" / "airtlab-replay.joblib")
    return susun_kekerasan(bus_train, airtlab, kb.bus(data / "putaran3", cfg), scfd)


def _per_source(rows: list[dict], test: list) -> dict:
    source = {r.item.name: r.item.source for r in test}
    out = {}
    for row in rows:
        s = out.setdefault(source[row["name"]], {"tp": 0, "pos": 0, "fp": 0, "neg": 0})
        if row["violent"]:
            s["pos"] += 1
            s["tp"] += bool(row["events"])
        else:
            s["neg"] += 1
            s["fp"] += bool(row["events"])
    return out


def _per_jam(model, cfg) -> float:
    """Alarm kekerasan keliru per jam pada kegiatan biasa sesi 17 (sama dengan laju_rendah.measure_rate)."""
    import tempfile
    from . import eksperimen_abc as abc
    original = abc.OUT
    with tempfile.TemporaryDirectory() as tmp:
        abc.OUT = Path(tmp)
        try:
            s = abc.session(cfg)
        finally:
            abc.OUT = original
    replay = s["replay"]
    probs = abc._violence_probs(model, [replay], cfg, True)[0]
    events = abc.violence_events(replay, probs, cfg, True, ack_after_s=3.0)
    return round(len([t for t in events if abc._in_normal(s, t)]) / s["normal_hours"], 2)


def ukur_kekerasan(model, test: list, cfg) -> dict:
    from . import latih_kekerasan_baru as kb
    rows = kb.evaluate(test, kb.predict(model, test, cfg), cfg)
    per = _per_source(rows, test)
    return {"tertangkap": sum(v["tp"] for v in per.values()), "positif": sum(v["pos"] for v in per.values()),
            "keliru_klip": sum(v["fp"] for v in per.values()), "negatif": sum(v["neg"] for v in per.values()),
            "keliru_per_jam": _per_jam(model, cfg), "per_data": per}


def _f1(m: dict) -> float:
    p = m["tp"] / (m["tp"] + m["fp"]) if m["tp"] + m["fp"] else 0.0
    r = m["tp"] / m["pos"] if m["pos"] else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def kekerasan(install: bool = False) -> int:
    import joblib
    import numpy as np
    from sahabat.config import MODELS_DIR
    from sahabat.profil_kamera import VIOLENCE_TINGGI
    from . import eksperimen_abc as abc
    from . import latih_kekerasan_baru as kb
    from .latih_negatif_jatuh import installed_sources
    from .putar import load_config_snapshot
    from .train_rf import _pair_probs
    from .violence_train import fit, stack
    cfg_train = load_config_snapshot(P3 / "config-baseline.json")      # resep model terpasang
    cfg = abc._cfg()                                                     # penilai aplikasi
    train, dev, test = violence_sets(DATA, cfg_train)
    print(f"latih tambahan {len(train)}, dev {len(dev)}, uji {len(test)}", flush=True)
    installed = joblib.load(MODELS_DIR / "violence_model.joblib")
    base_rows = [np.concatenate(parts) for parts in zip(*installed_sources(DATA, P3, cfg_train), strict=True)]
    items = [r.item for r in train]
    peaks = _pair_probs(installed, [it.clip for it in items])
    report = {"sekarang": {"dev": kb.metrics(kb.evaluate(dev, kb.predict(installed, dev, cfg), cfg), cfg)}}
    best = None
    for weight in BOBOT:
        file = ROOT / f"violence-tinggi-b{weight}.joblib"
        if file.is_file():
            model = joblib.load(file)
        else:
            extra = stack(items, weight, peaks)
            x, y, w = [np.concatenate((a, b)) for a, b in zip(base_rows, extra, strict=True)]
            print(f"bobot {weight}: {len(y)} baris latih", flush=True)
            model = fit(x, y, w)
            temporary = file.with_suffix(".part")
            joblib.dump(model, temporary, compress=3)
            temporary.replace(file)
        m = kb.metrics(kb.evaluate(dev, kb.predict(model, dev, cfg), cfg), cfg)
        report[f"b{weight}"] = {"dev": m, "f1_dev": round(_f1(m), 3)}
        print(f"bobot {weight} dev", m, flush=True)
        if best is None or _f1(m) > best[1]:
            best = (weight, _f1(m), file, model)
    weight, _, file, model = best
    sekarang, tinggi = ukur_kekerasan(installed, test, cfg), ukur_kekerasan(model, test, cfg)
    passed = lolos_tinggi(sekarang, tinggi)
    report.update(terpilih=f"b{weight}", uji_sekarang=sekarang, uji_tinggi=tinggi, lolos=passed,
                  tanggal=time.strftime("%Y-%m-%d"))
    installed_now = pasang(file, VIOLENCE_TINGGI, passed) if install else False
    report["dipasang"] = installed_now
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kekerasan.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"{'data':12} {'sekarang':>14} {'tinggi':>14}")
    for src in sekarang["per_data"]:
        a, b = sekarang["per_data"][src], tinggi["per_data"][src]
        print(f"{src:12} {a['tp']:>3}/{a['pos']:<3} kel {a['fp']:>3}/{a['neg']:<3}  {b['tp']:>3}/{b['pos']:<3} kel {b['fp']:>3}/{b['neg']:<3}")
    print(f"keliru/jam   {sekarang['keliru_per_jam']:>14} {tinggi['keliru_per_jam']:>14}")
    print(("LOLOS" if passed else "TIDAK LOLOS") + f" (kandidat b{weight})" + (", dipasang" if installed_now else ""))
    return 0


# ---- jatuh ----------------------------------------------------------------------------------

def susun_jatuh(train: list, dev: list, test: list) -> tuple[list, list, list]:
    """Klip uji (nama, atau subjek untuk CAUCAFall) tidak boleh ada di latih atau dev."""
    def keys(rows):
        return {(c.source, c.name) for c, _ in rows}, {(c.source, c.group) for c, _ in rows if c.source == "cauca"}
    test_names, test_groups = keys(test)
    for label, rows in (("latih", train), ("dev", dev)):
        names, groups = keys(rows)
        if names & test_names or groups & test_groups:
            raise ValueError(f"Data {label} bocor ke uji: {sorted(names & test_names)[:3]} {sorted(groups & test_groups)[:3]}")
    return train, dev, test


def fall_sets(data: Path = DATA, cfg=None) -> tuple[list, list, list]:
    """(latih tambahan, dev, uji tinggi): CAUCAFall subjek 1-6 / 7-8, uji = CAUCAFall 9-10 + Le2i (fall_test_clips)."""
    from . import eksperimen_abc as abc
    from .latih_jatuh_baru import caucafall
    return susun_jatuh(list(caucafall(data / "putaran3", cfg, set(range(1, 7)))),
                       list(caucafall(data / "putaran3", cfg, {7, 8})), abc.fall_test_clips(cfg))


def _fall_counts(stats: dict) -> tuple[int, int, int, int]:
    tp = sum(v["tp"] for v in stats.values())
    pos = sum(v["pos"] for v in stats.values())
    fp = sum(v["fp"] + v["early"] for v in stats.values())            # alarm sebelum jatuh juga keliru
    neg = sum(v["neg"] for v in stats.values())
    return tp, pos, fp, neg


def _fall_per_jam(models: dict, gates: dict, cfg) -> dict:
    """Alarm jatuh (model + aturan aplikasi) per jam pada kegiatan biasa sesi 17."""
    import tempfile
    from types import SimpleNamespace
    from . import eksperimen_abc as abc
    from .latih_jatuh_baru import model_events
    original = abc.OUT
    with tempfile.TemporaryDirectory() as tmp:
        abc.OUT = Path(tmp)
        try:
            s = abc.session(cfg)
        finally:
            abc.OUT = original
    events = model_events(SimpleNamespace(trace=s["trace"]), cfg, models, gates)
    return {name: round(len([e for e in ev if abc._in_normal(s, e["t"])]) / s["normal_hours"], 2)
            for name, ev in events.items()}


def _sesi16_tinggi(models: dict, gates: dict, cfg) -> dict:
    """Data penyetelan (aturan turun_bawah disetel dari sini): J4-J5 jatuh dan B17-B18 biasa, kamera tinggi."""
    from types import SimpleNamespace
    from .latih_jatuh_baru import model_events
    from .putar import PoseTrace
    tr = PoseTrace.load(SESI16 / "jejak_416.npz")
    t0 = float(tr.t[0])
    labels = json.loads((SESI16 / "label.json").read_text(encoding="utf-8"))
    takes = [(t0 + x["mulai"], t0 + x["selesai"], x["catatan"].startswith("J4-J5"))
             for x in labels if x["catatan"].startswith(("J4-J5", "B17-B18"))]
    events = model_events(SimpleNamespace(trace=tr), cfg, models, gates)
    out = {}
    for name, ev in events.items():
        hit = [any(a <= e["t"] <= b + 3.0 for e in ev) for a, b, _ in takes]
        out[name] = {"jatuh": f"{sum(h for h, (_, _, f) in zip(hit, takes) if f)}/{sum(f for _, _, f in takes)}",
                     "biasa_beralarm": f"{sum(h for h, (_, _, f) in zip(hit, takes) if not f)}/"
                                       f"{sum(not f for _, _, f in takes)}"}
    return out


def jatuh(install: bool = False) -> int:
    import joblib
    from sahabat.config import MODELS_DIR
    from sahabat.profil_kamera import FALL_TINGGI
    from . import eksperimen_abc as abc
    from .latih_jatuh_baru import fit_candidate, legacy, measure
    cfg = abc._cfg()
    train, dev, test = fall_sets(DATA, cfg)
    print(f"latih {len(train)}, dev {len(dev)}, uji {len(test)}", flush=True)
    installed = joblib.load(MODELS_DIR / "fall_model.joblib")
    gate = float(getattr(installed, "fall_rule_support_threshold_", 0.0))
    core, _ = legacy(DATA, cfg, P3)
    models, files = {"sekarang": installed}, {}
    for weight in BOBOT_JATUH:
        file = ROOT / f"fall-tinggi-b{weight}.joblib"
        if file.is_file():
            model = joblib.load(file)
        else:
            model = fit_candidate(core, train, "descent20", weight)
            model.fall_rule_support_threshold_ = gate
            temporary = file.with_suffix(".part")
            joblib.dump(model, temporary, compress=3)
            temporary.replace(file)
        models[f"b{weight}"], files[f"b{weight}"] = model, file
    gates = {name: gate for name in models}
    dev_result = measure(dev, cfg, models, gates)
    score = {name: (lambda c: c[0] - c[2])(_fall_counts(dev_result[name]["metrics"])) for name in files}
    chosen = max(files, key=lambda name: score[name])
    test_result = measure(test, cfg, {"sekarang": installed, chosen: models[chosen]}, gates)
    per_jam = _fall_per_jam({"sekarang": installed, chosen: models[chosen]}, gates, cfg)
    result = {}
    for name in ("sekarang", chosen):
        tp, pos, fp, neg = _fall_counts(test_result[name]["metrics"])
        result[name] = {"tertangkap": tp, "positif": pos, "keliru_klip": fp, "negatif": neg,
                        "keliru_per_jam": per_jam[name], "per_data": test_result[name]["metrics"]}
    passed = lolos_tinggi(result["sekarang"], result[chosen])
    installed_now = pasang(files[chosen], FALL_TINGGI, passed) if install else False
    report = {"tanggal": time.strftime("%Y-%m-%d"), "dev": {n: dev_result[n]["metrics"] for n in models},
              "terpilih": chosen, "uji_sekarang": result["sekarang"], "uji_tinggi": result[chosen],
              "sesi16_penyetelan": _sesi16_tinggi({"sekarang": installed, chosen: models[chosen]}, gates, cfg),
              "lolos": passed, "dipasang": installed_now}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jatuh.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print()
    for name in ("sekarang", chosen):
        r = result[name]
        print(f"{name:9} tertangkap {r['tertangkap']}/{r['positif']}, keliru {r['keliru_klip']}/{r['negatif']}, "
              f"keliru/jam {r['keliru_per_jam']}, sesi16 {report['sesi16_penyetelan'][name]}")
    print(("LOLOS" if passed else "TIDAK LOLOS") + f" (kandidat {chosen})" + (", dipasang" if installed_now else ""))
    return 0


# ---- rekaman sekolah -------------------------------------------------------------------------

POS_BEFORE_S, POS_AFTER_S = 8.0, 0.5      # tanda ditekan sesudah adegan: positif 8 dtk sebelum sampai 0,5 dtk sesudah
BIASA_BEFORE_S = 12.0                     # tanda "biasa": 12 dtk sebelumnya negatif (sama dengan sesi 16)
BIASA_CHUNK_S = 60.0                      # rekaman kegiatan biasa dibagi per menit (satuan lipatan validasi silang)
SEKOLAH_REPORT = REPO / "docs" / "uji"


def label_dari_tanda(marks: list[tuple[float, str]], jenis: str, t0: float, durasi: float) -> list[dict]:
    """Rentang berlabel (detik sejak frame pertama, format label.json / violence_train.session_items).

    Rekaman "adegan": tiap tanda kekerasan/jatuh = positif 8 dtk sebelum tanda sampai 0,5 dtk sesudah; tanda biasa =
    negatif 12 dtk sebelumnya. Jatuh adalah negatif untuk model kekerasan. Rekaman "biasa": seluruhnya negatif, per menit.
    """
    if jenis == "biasa":
        rows, start = [], 0.0
        while start < durasi - 1e-6:
            end = min(start + BIASA_CHUNK_S, durasi)
            rows.append({"mulai": round(start, 3), "selesai": round(end, 3), "kekerasan": False, "jatuh": False,
                         "grup": len(rows)})
            start = end
        return rows
    rows, prev_end = [], 0.0
    for t, label in sorted(marks):
        rel = t - t0
        if label in ("kekerasan", "jatuh"):
            lo, hi = rel - POS_BEFORE_S, rel + POS_AFTER_S
        elif label == "biasa":
            lo, hi = rel - BIASA_BEFORE_S, rel
        else:
            continue
        # Rentang tidak boleh menumpuk rentang sebelumnya: frame yang sama tidak boleh berlabel ganda (dan masuk dua
        # lipatan validasi silang). Sama dengan match_marks yang membatasi pencarian pada tanda sebelumnya.
        lo, hi = max(0.0, lo, prev_end), min(durasi, hi)
        if hi <= lo:
            continue
        rows.append({"mulai": round(lo, 3), "selesai": round(hi, 3),
                     "kekerasan": label == "kekerasan", "jatuh": label == "jatuh", "grup": len(rows)})
        prev_end = hi
    return rows


def _meta(folder: Path) -> dict:
    try:
        data = json.loads((Path(folder) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {"jenis": data.get("jenis", "adegan"), "kamera": str(data.get("kamera", Path(folder).name)),
            "posisi": data.get("posisi", "dada")}


def ukur_sesi(folder: Path, models: dict, cfg) -> dict:
    """Alarm aplikasi (Engine sama dengan aplikasi) per model pada satu rekaman: kejadian tertangkap per jenis tanda,
    ulangan biasa yang beralarm, dan untuk rekaman kegiatan biasa: alarm per jam."""
    from sahabat.recorder import read_marks
    from .putar import load_or_trace, match_marks, replay
    tr = load_or_trace(Path(folder), 416, cfg.rate_hz)
    meta, marks = _meta(folder), read_marks(folder)
    hours = (float(tr.t[-1]) - float(tr.t[0])) / 3600.0 if len(tr.t) > 1 else 0.0
    out = {"meta": meta, "jam": round(hours, 3)}
    for name, (violence, fall) in models.items():
        alarms = replay(cfg, tr, fall, violence)
        if meta["jenis"] == "biasa":
            out[name] = {"alarm_biasa": len(alarms)}
            continue
        rows = match_marks(alarms, marks)
        count = {}
        for label in ("jatuh", "kekerasan", "biasa"):
            mine = [r for r in rows if r.label == label]
            count[label] = [sum(r.ok for r in mine), len(mine)]
        out[name] = count
    return out


def _total(results: list[dict], name: str) -> dict:
    tot = {"jatuh": [0, 0], "kekerasan": [0, 0], "biasa": [0, 0], "alarm_biasa": 0, "jam_biasa": 0.0}
    for r in results:
        got = r.get(name)
        if got is None:
            continue
        if r["meta"]["jenis"] == "biasa":
            tot["alarm_biasa"] += got["alarm_biasa"]
            tot["jam_biasa"] += r["jam"]
        else:
            for label in ("jatuh", "kekerasan", "biasa"):
                tot[label][0] += got[label][0]
                tot[label][1] += got[label][1]
    return tot


def _laporan_sekolah(results: list[dict], models: dict, latih: dict | None) -> str:
    lines = [f"# Rekaman sekolah: model sekarang dan model kamera tinggi, {time.strftime('%Y-%m-%d')}", "",
             "Dibuat oleh `python -m latih.kamera_tinggi sekolah`. Alarm dihitung dengan mesin yang sama dengan aplikasi.",
             "", "## Rekaman", "", "| Rekaman | Kamera | Posisi | Jenis | Durasi |", "|---|---|---|---|---|"]
    for r in results:
        m = r["meta"]
        lines.append(f"| {r['folder']} | {m['kamera']} | {m['posisi']} | {m['jenis']} | {r['jam'] * 60:.1f} menit |")
    lines += ["", "## Hasil", "", "| Model | Tertangkap | Ulangan biasa tanpa alarm | Alarm keliru per jam |",
              "|---|---|---|---|"]
    for name in models:
        t = _total(results, name)
        per_jam = f"{t['alarm_biasa'] / t['jam_biasa']:.1f}" if t["jam_biasa"] > 0 else "belum ada rekaman kegiatan biasa"
        lines.append(f"| {name} | kekerasan {t['kekerasan'][0]}/{t['kekerasan'][1]}, "
                     f"jatuh {t['jatuh'][0]}/{t['jatuh'][1]} | {t['biasa'][0]}/{t['biasa'][1]} | {per_jam} |")
    if "tinggi" not in models:
        lines += ["", "Model kamera tinggi belum terpasang (belum lolos uji), jadi hanya model sekarang yang diukur."]
    if latih is not None:
        lines += ["", "## Latih ulang model kekerasan tinggi (validasi silang 3 lipatan)", "",
                  "Tiap ulangan dinilai model yang tidak melihatnya.", "",
                  "| Model | Kekerasan tertangkap | Klip biasa beralarm | Alarm keliru per jam |", "|---|---|---|---|"]
        for name in ("sekarang", "tinggi"):
            v = latih[name]
            lines.append(f"| {name} | {v['tertangkap']}/{v['positif']} | {v['keliru_klip']}/{v['negatif']} | "
                         f"{v['keliru_per_jam']} |")
        lines += ["", "**Lolos.**" if latih["lolos"] else "**Tidak lolos**: model tinggi tidak dipasang.",
                  "Dipasang ke models/." if latih.get("dipasang") else ""]
        if latih["lolos"] and not latih.get("jam_biasa"):
            lines.append("Belum ada rekaman kegiatan biasa, jadi alarm keliru per jam belum terukur: model tidak "
                         "boleh dipasang sebelum itu.")
    lines += ["", "Angka ini dari satu kunjungan dan satu sudut kamera, bukan akurasi umum."]
    return "\n".join(lines) + "\n"


def lipatan_sekolah(items: list, replays: list):
    """Lipatan validasi silang per ulangan: (item latih, (replay, biasa?) yang dinilai) untuk k = 0, 1, 2.

    `session_items` memberi Item nomor grup WEBCAM_GROUP_BASE + g, sedangkan replay penilaian memakai g. Lipatan
    dihitung dari g yang sama, dan ulangan yang dinilai tidak pernah ada di data latih lipatan itu.
    """
    from .violence_train import WEBCAM_GROUP_BASE
    for k in range(3):
        train_items = [it for it in items if (it.group - WEBCAM_GROUP_BASE) % 3 != k]
        held = [(rep, biasa) for rep, biasa in replays if rep.item.group % 3 == k]
        trained = {it.group - WEBCAM_GROUP_BASE for it in train_items}
        if trained & {rep.item.group for rep, _ in held}:
            raise ValueError(f"Lipatan {k}: ulangan yang dinilai ikut dilatih.")
        yield train_items, held


def boleh_pasang_sekolah(lolos: bool, jam_biasa: float) -> bool:
    """Tanpa rekaman kegiatan biasa, alarm keliru per jam tidak terukur, jadi model tidak boleh dipasang."""
    return lolos and jam_biasa > 0


def _latih_sekolah(folders: list[Path], install: bool) -> dict:
    """Validasi silang 3 lipatan: model kekerasan tinggi (resep kandidat kekerasan di atas + rekaman sekolah) vs model terpasang."""
    import joblib
    import numpy as np
    from sahabat.config import MODELS_DIR
    from sahabat.profil_kamera import VIOLENCE_TINGGI
    from sahabat.recorder import read_marks
    from . import eksperimen_abc as abc
    from . import latih_kekerasan_baru as kb
    from .latih_negatif_jatuh import installed_sources
    from .putar import load_config_snapshot, load_or_trace
    from .train_rf import _pair_probs
    from .violence_train import fit, session_items, stack
    cfg_train, cfg = load_config_snapshot(P3 / "config-baseline.json"), abc._cfg()
    chosen = json.loads((OUT / "kekerasan.json").read_text(encoding="utf-8")).get("terpilih", "b5") \
        if (OUT / "kekerasan.json").is_file() else "b5"
    weight = int(chosen.lstrip("b"))
    installed = joblib.load(MODELS_DIR / "violence_model.joblib")
    base = [np.concatenate(p) for p in zip(*installed_sources(DATA, P3, cfg_train), strict=True)]
    high, _, _ = violence_sets(DATA, cfg_train)
    high_items = [r.item for r in high]
    high_rows = stack(high_items, weight, _pair_probs(installed, [it.clip for it in high_items]))
    items, replays, biasa_hours = [], [], 0.0
    for n, folder in enumerate(folders):
        tr = load_or_trace(Path(folder), 416, cfg.rate_hz)
        meta = _meta(folder)
        t0, durasi = float(tr.t[0]), float(tr.t[-1]) - float(tr.t[0])
        ranges = label_dari_tanda(read_marks(folder), meta["jenis"], t0, durasi)
        for r in ranges:
            r["grup"] = n * 1000 + r["grup"]
        items += session_items(tr, ranges, cfg_train)
        if meta["jenis"] == "biasa":
            biasa_hours += durasi / 3600.0
        seconds = np.asarray(tr.t, dtype=float) - t0
        for r in ranges:
            inside = np.flatnonzero((seconds >= r["mulai"] - 1e-6) & (seconds <= r["selesai"] + 1e-6))
            frames = [(float(tr.t[i]), tr.people[i]) for i in inside]
            rep = kb.prepare(f"{Path(folder).name}/{r['grup']}", "sekolah", r["kekerasan"], r["grup"], "test",
                             frames, (tr.frame_h, tr.frame_w), cfg)
            replays.append((rep, meta["jenis"] == "biasa"))
    if not items:
        raise ValueError("Rekaman sekolah tidak berisi rentang berlabel.")
    rows = {"sekarang": [], "tinggi": []}
    for train_items, held in lipatan_sekolah(items, replays):
        if not held:
            continue
        extra = stack(train_items, 10.0) if train_items else None
        parts = [base, high_rows] + ([extra] if extra is not None else [])
        model = fit(*[np.concatenate(p) for p in zip(*parts, strict=True)])
        test = [rep for rep, _ in held]
        for name, m in (("sekarang", installed), ("tinggi", model)):
            for row, (_, biasa) in zip(kb.evaluate(test, kb.predict(m, test, cfg), cfg), held, strict=True):
                rows[name].append((row, biasa))
    result = {}
    for name, got in rows.items():
        pos = [r for r, _ in got if r["violent"]]
        neg = [r for r, _ in got if not r["violent"]]
        alarms_biasa = sum(len(r["events"]) for r, biasa in got if biasa)
        result[name] = {"tertangkap": sum(bool(r["events"]) for r in pos), "positif": len(pos),
                        "keliru_klip": sum(bool(r["events"]) for r in neg), "negatif": len(neg),
                        "keliru_per_jam": round(alarms_biasa / biasa_hours, 2) if biasa_hours else 0.0}
    result["lolos"] = lolos_tinggi(result["sekarang"], result["tinggi"])
    result["dipasang"] = False
    result["jam_biasa"] = round(biasa_hours, 3)
    if install and not boleh_pasang_sekolah(result["lolos"], biasa_hours):
        print("Tidak dipasang: " + ("belum ada rekaman kegiatan biasa (alarm keliru per jam belum terukur)."
                                    if result["lolos"] else "tidak lolos gerbang."))
    if install and boleh_pasang_sekolah(result["lolos"], biasa_hours):
        final = fit(*[np.concatenate(p) for p in zip(base, high_rows, stack(items, 10.0), strict=True)])
        file = ROOT / "violence-tinggi-sekolah.joblib"
        joblib.dump(final, file, compress=3)
        result["dipasang"] = pasang(file, VIOLENCE_TINGGI, True)
    return result


def sekolah(folders: list[Path], latih: bool = False, install: bool = False, laporan: Path | None = None) -> int:
    from sahabat.profil_kamera import model_untuk
    from . import eksperimen_abc as abc
    cfg = abc._cfg()
    now, high = model_untuk("dada"), model_untuk("tinggi")
    models = {"sekarang": (now.violence, now.fall)}
    if (high.violence, high.fall) != (now.violence, now.fall):
        models["tinggi"] = (high.violence, high.fall)
    results = []
    for folder in folders:
        r = ukur_sesi(Path(folder), models, cfg)
        r["folder"] = Path(folder).name
        results.append(r)
        print(r, flush=True)
    trained = _latih_sekolah([Path(f) for f in folders], install) if latih else None
    target = laporan or SEKOLAH_REPORT / f"hasil-kamera-tinggi-sekolah-{time.strftime('%Y%m%d')}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_laporan_sekolah(results, models, trained), encoding="utf-8")
    print("Laporan:", target)
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["jejak"]:
        return jejak()
    if argv[:1] == ["kekerasan"]:
        return kekerasan("--pasang" in argv)
    if argv[:1] == ["jatuh"]:
        return jatuh("--pasang" in argv)
    if argv[:1] == ["sekolah"]:
        folders = [Path(x) for x in argv[1:] if not x.startswith("--")]
        if not folders:
            print("Sebutkan folder rekaman, misalnya data/rekaman/20270110-090000")
            return 2
        return sekolah(folders, "--latih" in argv, "--pasang" in argv)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
