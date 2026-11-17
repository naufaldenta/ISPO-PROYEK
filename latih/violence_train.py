"""Melatih penilai kekerasan yang dipakai aplikasi: klip dua-orang NTU RGB+D ditambah video sungguhan.

    python -m latih.ntu_build <ntu120_2d.pkl> <cache.npz>      (sekali; menghitung fitur tiap klip NTU)
    python -m latih.violence_train <cache.npz> [--ut <folder UT-Interaction>] [--webcam <folder sesi>] [--bobot 10]
                                          [--output-dir <folder kandidat>]

Positif: pukul/tampar, tendang, dorong. Negatif: aksi ramah (salaman, pelukan, tos, tepuk punggung, ...).

NTU (kerangka HRNet, kamera Kinect) memberi jumlah dan ragam sudut pandang. UT-Interaction memberi kemiripan
dengan aplikasi: videonya dilewatkan YOLO-pose + ByteTrack yang sama, jadi fiturnya memuat getar dan titik
tertukar yang sama dengan yang dilihat aplikasi. Jendela dari video diberi bobot karena jumlahnya jauh lebih
sedikit. Diukur 15 November 2026: model yang hanya dilatih dari NTU tidak bekerja di video sungguhan.

Label hanya ada per klip. Putaran pertama menganggap semua jendela bergerak dari klip kekerasan positif,
termasuk saat mendekat dan menjauh yang sama saja dengan klip ramah; akibatnya peluang di video sungguhan
menggelembung dan tidak ada satu ambang yang cocok untuk kedua sumber. Putaran kedua hanya memakai
PEAK_ROWS jendela berpeluang tertinggi per klip kekerasan (dipilih oleh model putaran pertama yang tidak
melihat klip itu) sebagai positif; jendela lain dari klip kekerasan dibuang, bukan dijadikan negatif.

--webcam: rekaman webcam sendiri yang sudah dijadikan jejak titik tubuh (jejak_416.npz) dan diberi label
rentang waktu (label.json: [{"mulai", "selesai", "kekerasan", "grup", "catatan"}], detik sejak frame pertama).
Rentang "biasa" adalah contoh negatif dari kondisi pakai yang sebenarnya; videonya sendiri tidak perlu disimpan.

Yang dilaporkan adalah berapa klip yang memunculkan ALARM, per aksi, dengan konfirmasi yang sama seperti
aplikasi: dari validasi silang (tiap klip dinilai model yang tidak pernah melihat pemeran NTU atau video asal
UT-nya) dan dari uji lintas-set (model tidak pernah melihat set itu sama sekali).
Rentang aksi webcam yang ditandai waktu memakai semua jendela bergerak; pemilihan puncak hanya untuk
NTU/UT yang berlabel seluruh klip. Model yang disimpan dilatih dari semua klip.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

from sahabat.config import MODELS_DIR, Config
from sahabat.pair_features import MOTION_FEATURES

from .ntu import FRIENDLY, VIOLENT
from .ntu_build import load_cache
from .train_rf import _pair_probs, clip_alarm, folds_by_clip
from .windows import _MOTION, ClipPairs, pairs_from_frames, training_rows

MODEL_PATH = MODELS_DIR / "violence_model.joblib"
REPORT_PATH = MODELS_DIR / "violence_model.laporan.txt"
THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95)
CONFIRMS = ((3, 5), (4, 6), (6, 10))
PEAK_ROWS = 10                 # putaran kedua: jendela puncak per klip kekerasan yang dipakai sebagai positif
MOTION_MIN = 1.0
ROW_STRIDE = 2                 # jendela NTU berurutan tumpang-tindih 90%: ambil selang-seling untuk melatih
VIDEO_WEIGHT = 10.0            # bobot jendela video sungguhan terhadap jendela NTU
IMGSZ = 416
UT_GROUP_BASE = 1000           # nomor grup video UT, supaya tidak bertabrakan dengan nomor pemeran NTU
WEBCAM_GROUP_BASE = 2000       # nomor grup rentang rekaman webcam sendiri


def action_name(action: int) -> str:
    return VIOLENT.get(action) or FRIENDLY.get(action) or f"aksi {action}"


def alarm_rates(hit: np.ndarray, actions: np.ndarray) -> dict[int, tuple[int, int]]:
    """{aksi: (klip beralarm, jumlah klip)}"""
    return {int(a): (int(hit[actions == a].sum()), int((actions == a).sum())) for a in sorted(set(actions.tolist()))}


def fit(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                          class_weight="balanced", random_state=0).fit(X, y, sample_weight=sample_weight)


def trace_pairs(tr, cfg: Config) -> ClipPairs:
    """Jejak YOLO-pose satu klip -> fitur pasangan per putaran, lewat History yang sama dengan aplikasi."""
    return pairs_from_frames(list(zip((float(t) for t in tr.t), tr.people)), cfg)


@dataclasses.dataclass
class Item:
    source: str            # ntu | ut | webcam
    name: str              # nama aksi
    violent: bool
    group: int             # pemeran NTU, atau UT_GROUP_BASE + set*100 + nomor video
    clip: ClipPairs


def ntu_items(cache: Path) -> list[Item]:
    clips, meta = load_cache(cache)
    return [Item("ntu", action_name(m["action"]), m["action"] in VIOLENT, int(m["performer"]), c)
            for c, m in zip(clips, meta)]


def ut_items(root: Path, cfg: Config) -> list[Item]:
    from .utinteraction import build_traces, load_index
    index = {c.name: c for c in load_index(root)}
    items = []
    for (name, _phase), tr in sorted(build_traces(Path(root), IMGSZ, cfg.rate_hz).items()):
        c = index[name]
        items.append(Item("ut", c.label, c.violent, UT_GROUP_BASE + 100 * c.set_no + c.sequence, trace_pairs(tr, cfg)))
    return items


def peak_rows(clip: ClipPairs, first: np.ndarray, k: int = PEAK_ROWS) -> np.ndarray:
    """Fitur k jendela berpeluang tertinggi di satu klip kekerasan (arah yang bergerak saja)."""
    n_feat = len(clip.rows[0][3]) if clip.rows else 0
    top = np.argsort(-np.asarray(first), kind="stable")[:k]
    feats = [f for j in top for f in (clip.rows[j][3], clip.rows[j][4])
             if max(f[i] for i in _MOTION) >= MOTION_MIN]
    return np.array(feats, dtype=np.float32).reshape(-1, n_feat)


def session_items(tr, ranges: list[dict], cfg: Config) -> list[Item]:
    """Satu Item per rentang berlabel dari satu rekaman webcam.

    Seluruh rekaman diputar sekali lewat History (riwayat tidak diputus di awal rentang), lalu baris fiturnya
    dibagi menurut rentang. `ranges`: mulai/selesai dalam detik sejak frame pertama, kekerasan (bool), grup (int).
    """
    whole = trace_pairs(tr, cfg)
    if len(tr.t) == 0:
        return []
    seconds = np.asarray(tr.t, dtype=float) - float(tr.t[0])
    items = []
    for r in ranges:
        inside = np.flatnonzero((seconds >= r["mulai"] - 1e-6) & (seconds <= r["selesai"] + 1e-6))
        place = {int(frame): k for k, frame in enumerate(inside)}
        rows = [(place[f], a, b, ab, ba) for f, a, b, ab, ba in whole.rows if f in place]
        clip = ClipPairs([float(whole.t[i]) for i in inside], rows)
        violent = bool(r["kekerasan"])
        items.append(Item("webcam", "kekerasan" if violent else "biasa", violent,
                          WEBCAM_GROUP_BASE + int(r.get("grup", 0)), clip))
    return items


def webcam_items(folder: Path, cfg: Config) -> list[Item]:
    from .putar import PoseTrace
    folder = Path(folder)
    ranges = json.loads((folder / "label.json").read_text(encoding="utf-8"))
    return session_items(PoseTrace.load(folder / f"jejak_{IMGSZ}.npz"), ranges, cfg)


def stack(items: list[Item], video_weight: float, first: list[np.ndarray] | None = None):
    """(X, y, bobot). `first`: peluang putaran pertama per klip; bila diberi, positif = jendela puncak saja."""
    X, y, w = [], [], []
    for n, it in enumerate(items):
        # Webcam memiliki batas waktu aksi yang ditandai langsung, bukan label
        # seluruh klip. Jangan buang aksi singkat yang belum dikenal model awal.
        if first is not None and it.violent and it.source != "webcam" and it.clip.rows:
            rows = peak_rows(it.clip, first[n])
        else:
            rows = training_rows(it.clip, int(it.violent), MOTION_MIN, ROW_STRIDE if it.source == "ntu" else 1)
        X.append(rows)
        y.append(np.full(len(rows), int(it.violent)))
        w.append(np.full(len(rows), 1.0 if it.source == "ntu" else video_weight))
    return np.concatenate(X), np.concatenate(y), np.concatenate(w)


def oof_probs(items: list[Item], video_weight: float, first: list[np.ndarray] | None = None,
              log=print) -> list[np.ndarray]:
    """Peluang per baris untuk tiap klip, dari model yang tidak melihat grup klip itu."""
    probs: list[np.ndarray] = [np.zeros(0)] * len(items)
    groups = np.array([it.group for it in items])
    for k, (train, test) in enumerate(folds_by_clip(groups), 1):
        started = time.monotonic()
        model = fit(*stack([items[i] for i in train], video_weight,
                           None if first is None else [first[i] for i in train]))
        for i, p in zip(test, _pair_probs(model, [items[i].clip for i in test])):
            probs[i] = p
        log(f"  lipatan {k} selesai ({time.monotonic() - started:.0f} dtk)")
    return probs


def alarms(items: list[Item], probs: list[np.ndarray], cfg: Config) -> np.ndarray:
    return np.array([clip_alarm(it.clip, p, cfg) for it, p in zip(items, probs)])


def motion_filtered(items: list[Item], probs: list[np.ndarray], cfg: Config) -> list[np.ndarray]:
    """Peluang konfirmasi setelah penyaring pasangan diam yang dipakai aplikasi."""
    filtered = []
    for item, p in zip(items, probs, strict=True):
        if not len(p):
            filtered.append(p)
            continue
        rows = np.asarray([(ab, ba) for _, _, _, ab, ba in item.clip.rows])
        active = rows[:, :, MOTION_FEATURES].max(axis=(1, 2)) >= cfg.violence_min_motion
        filtered.append(np.where(active, p, 0.0))
    return filtered


def rate(hit: np.ndarray, sel: np.ndarray) -> str:
    return f"{100.0 * hit[sel].mean():5.1f}%" if sel.any() else "    - "


def per_action_lines(items: list[Item], hit: np.ndarray, source: str) -> list[str]:
    lines = []
    names = sorted({(not it.violent, it.name) for it in items if it.source == source})
    for not_violent, name in names:
        sel = np.array([it.source == source and it.name == name for it in items])
        scored = sum(bool(it.clip.rows) for it, s in zip(items, sel) if s)
        kind = "ramah" if not_violent else "KEKERASAN"
        lines.append(f"  {name:28s} {kind:9s} {int(hit[sel].sum()):4d}/{int(sel.sum()):<4d} ({100 * hit[sel].mean():5.1f}%)"
                     + ("" if scored == sel.sum() else f"   pasangan dinilai di {scored} klip"))
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
    weight = float(opts.get("bobot", VIDEO_WEIGHT))
    started = time.monotonic()
    cfg = Config()
    items = ntu_items(Path(args[0]))
    print(f"NTU: {len(items)} klip ({time.monotonic() - started:.0f} dtk)", flush=True)
    if "ut" in opts:
        items += ut_items(Path(opts["ut"]), cfg)
    if "webcam" in opts:
        items += webcam_items(Path(opts["webcam"]), cfg)
    src = np.array([it.source for it in items])
    violent = np.array([it.violent for it in items])
    names = np.array([it.name for it in items])
    has_ut = bool((src == "ut").any())
    print(f"semua sumber: {len(items)} klip ({time.monotonic() - started:.0f} dtk)", flush=True)

    print("putaran pertama (semua jendela bergerak dari klip kekerasan dianggap positif):", flush=True)
    first = oof_probs(items, weight)
    X, y, w = stack(items, weight, first)
    default = (cfg.violence_threshold, cfg.confirm_k, cfg.confirm_n)
    lines = [
        f"sumber: {args[0]}" + (f", ut={opts['ut']}" if has_ut else "")
        + (f", webcam={opts['webcam']}" if "webcam" in opts else ""),
        f"batas jarak pasangan: {cfg.pair_max_dist:g} ukuran badan",
        f"klip NTU: kekerasan={int((violent & (src == 'ntu')).sum())}  ramah={int((~violent & (src == 'ntu')).sum())}",
        f"putaran video UT-Interaction: kekerasan={int((violent & (src == 'ut')).sum())}"
        f"  ramah={int((~violent & (src == 'ut')).sum())} (termasuk 'menunjuk' yang hanya satu orang)",
        f"rentang rekaman webcam sendiri: kekerasan={int((violent & (src == 'webcam')).sum())}"
        f"  biasa={int((~violent & (src == 'webcam')).sum())}",
        f"jendela latih: kekerasan={int((y == 1).sum())} ({PEAK_ROWS} puncak per klip NTU/UT; webcam semua gerak berlabel)"
        f"  ramah={int((y == 0).sum())}; bobot jendela video {weight:g}",
        "",
    ]
    del X, y, w

    def summary(hit: np.ndarray, source: str) -> str:
        m = src == source
        if source == "ntu":
            return f"NTU kekerasan {rate(hit, m & violent)}  ramah {rate(hit, m & ~violent)}"
        return (f"UT kekerasan {rate(hit, m & violent)}  pelukan {rate(hit, m & (names == 'pelukan'))}"
                f"  salaman {rate(hit, m & (names == 'salaman'))}")

    if has_ut:
        lines.append(f"Uji lintas-set pada bawaan aplikasi (ambang {default[0]}, {default[1]} dari {default[2]}):")
        for held in ("ut", "ntu"):
            train_idx = [i for i, it in enumerate(items) if it.source != held]
            test_idx = [i for i, it in enumerate(items) if it.source == held]
            sources = "+".join(sorted({items[i].source for i in train_idx}))
            title = f"dilatih dari {sources}, diuji pada {held}"
            model = fit(*stack([items[i] for i in train_idx], weight, [first[i] for i in train_idx]))
            p = _pair_probs(model, [items[i].clip for i in test_idx])
            hit = np.zeros(len(items), dtype=bool)
            test_items = [items[i] for i in test_idx]
            hit[test_idx] = alarms(test_items, motion_filtered(test_items, p, cfg), cfg)
            lines.append(f"- {title}: {summary(hit, held)}")
            print(lines[-1], flush=True)
        lines.append("")

    print("putaran kedua (positif = jendela puncak), validasi silang semua sumber:", flush=True)
    probs = oof_probs(items, weight, first)
    filtered_probs = motion_filtered(items, probs, cfg)
    lines += ["Klip yang memunculkan ALARM kekerasan (validasi silang 5 lipatan; model dilatih dari semua sumber):",
              "  ambang  konfirmasi | NTU: kekerasan  ramah | UT-Interaction: kekerasan  pelukan  salaman"]
    hits = {}
    for thr, k, n in sorted({(t, k, n) for t in THRESHOLDS for k, n in CONFIRMS} | {default}):
        hit = alarms(items, filtered_probs,
                     dataclasses.replace(cfg, violence_threshold=thr, confirm_k=k, confirm_n=n))
        hits[(thr, k, n)] = hit
        ntu, ut = src == "ntu", src == "ut"
        mark = "  <- bawaan aplikasi" if (thr, k, n) == default else ""
        lines.append(f"  {thr:<6}  {k} dari {n:<3}   |      {rate(hit, ntu & violent)}   {rate(hit, ntu & ~violent)} |"
                     f"                  {rate(hit, ut & violent)}   {rate(hit, ut & (names == 'pelukan'))}"
                     f"   {rate(hit, ut & (names == 'salaman'))}{mark}")
    if (src == "webcam").any():
        cam = src == "webcam"
        lines += ["", "Rentang rekaman webcam sendiri yang memunculkan ALARM (validasi silang; tiap blok waktu dinilai model yang",
                  "tidak melihat blok itu). Rentang 'biasa' yang beralarm = alarm keliru:",
                  "  ambang  konfirmasi | kekerasan   biasa"]
        for (thr, k, n), h in sorted(hits.items()):
            mark = "  <- bawaan aplikasi" if (thr, k, n) == default else ""
            lines.append(f"  {thr:<6}  {k} dari {n:<3}   |  {int(h[cam & violent].sum())}/{int((cam & violent).sum())}"
                         f"        {int(h[cam & ~violent].sum())}/{int((cam & ~violent).sum())}{mark}")
    hit = hits[default]
    lines += ["", f"Per aksi pada bawaan aplikasi (ambang {default[0]}, {default[1]} dari {default[2]}), NTU:"]
    lines += per_action_lines(items, hit, "ntu")
    if has_ut:
        lines += ["", "Per aksi pada bawaan aplikasi, UT-Interaction (video sungguhan lewat YOLO-pose):"]
        lines += per_action_lines(items, hit, "ut")
    lines += [
        "",
        "Catatan: titik tubuh NTU berasal dari HRNet dan kamera Kinect di tiga ketinggian; UT-Interaction direkam",
        "dari satu sudut pandang (samping, agak atas). Webcam yang ikut dilatih bukan uji independen; blok waktu",
        "dari sesi sama masih berbagi pemeran dan latar. Uji sesi/pemeran baru tetap diperlukan.",
        "Jendela puncak dipilih oleh model putaran pertama yang lipatannya sama; model itu tidak melihat klip yang",
        "dipilihkannya, tetapi melihat klip uji lipatan lain, jadi angka di atas sedikit optimistis.",
    ]

    final = fit(*stack(items, weight, first))
    output = Path(opts.get("output-dir", MODELS_DIR))
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / MODEL_PATH.name
    joblib.dump(final, model_path, compress=3)
    (output / REPORT_PATH.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    oof_dir = output if "output-dir" in opts else Path(args[0]).parent
    np.savez_compressed(oof_dir / "oof_kekerasan.npz", probs=np.concatenate(probs),
                        first=np.concatenate(first), offsets=np.cumsum([0] + [len(p) for p in probs]))
    print("\n".join(lines))
    print(f"\nmodel disimpan: {model_path}  ({time.monotonic() - started:.0f} dtk)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
