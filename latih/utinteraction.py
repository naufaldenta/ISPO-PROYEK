"""Mengukur penilai kekerasan pada video sungguhan (UT-Interaction) lewat jalur aplikasi yang asli.

    python -m latih.utinteraction <folder berisi segmented_set1/ dan segmented_set2/> [resolusi]

120 klip dua orang yang sudah dipotong per aksi: dorong, tendang, pukul (kekerasan) dan salaman, pelukan,
menunjuk (ramah). Direkam dari samping agak atas, seluruh badan terlihat, 30 gambar per detik. Set 1 di
lapangan parkir, set 2 di rumput saat berangin (kamera sedikit bergoyang). Klip "menunjuk" hanya memuat
satu orang, jadi tidak ada pasangan yang dinilai.

Sumber: Ryoo & Aggarwal, "UT-Interaction Dataset, ICPR contest on Semantic Description of Human Activities
(SDHA)", 2010, http://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html. Syarat pakai: menyebut sumber.

Nama berkas: <nomor>_<urutan video>_<kelas>.avi. Arti nomor kelas dicek dari gambarnya (14 November 2026).
Klip tidak dipakai melatih; hanya untuk mengukur. Tiap klip dinilai dengan tiga frame awal.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sahabat.config import MODELS_DIR, Config

from .putar import Alarm, PoseTrace, replay_log, trace, video_ticks

CLASSES: dict[int, tuple[str, bool]] = {       # nomor kelas -> (nama, kekerasan?)
    0: ("salaman", False), 1: ("pelukan", False), 2: ("tendang", True),
    3: ("menunjuk", False), 4: ("pukul", True), 5: ("dorong", True),
}
PHASES = (0, 1, 2)


@dataclass(frozen=True)
class Clip:
    name: str               # set1-13-seq03-tendang
    set_no: int
    sequence: int           # video asal; orang dan latarnya sama dalam satu urutan
    label: str
    violent: bool
    path: Path


def load_index(root: Path) -> list[Clip]:
    clips = []
    for set_no in (1, 2):
        for path in sorted((Path(root) / f"segmented_set{set_no}").glob("*.avi")):
            parts = path.stem.split("_")
            if len(parts) != 3 or not all(p.isdigit() for p in parts) or int(parts[2]) not in CLASSES:
                continue
            index, sequence, cls = (int(p) for p in parts)
            label, violent = CLASSES[cls]
            clips.append(Clip(f"set{set_no}-{index:02d}-seq{sequence:02d}-{label}", set_no, sequence, label,
                              violent, path))
    return sorted(clips, key=lambda c: c.name)


def build_traces(root: Path, imgsz: int, rate_hz: float, phases=PHASES) -> dict[tuple[str, int], PoseTrace]:
    """YOLO-pose untuk semua klip dan fase; hasilnya disimpan supaya menyetel ambang tidak mengulang inferensi."""
    cache = Path(root) / f"jejak_{imgsz}"
    cache.mkdir(exist_ok=True)
    pose, out, started = None, {}, time.monotonic()
    clips = load_index(root)
    for n, clip in enumerate(clips, 1):
        fresh = False
        for phase in phases:
            saved = cache / f"{clip.name}.p{phase}.npz"
            if saved.is_file():
                out[clip.name, phase] = PoseTrace.load(saved)
                continue
            if pose is None:
                from sahabat.pose import YoloPose
                pose = YoloPose(imgsz)
            pose.reset()
            tr = trace(video_ticks(clip.path, rate_hz, phase), pose)
            tr.save(saved)
            out[clip.name, phase] = tr
            fresh = True
        if fresh and n % 10 == 0:
            print(f"  {n}/{len(clips)} klip  ({time.monotonic() - started:.0f} dtk)", flush=True)
    return out


@dataclass
class PairResult:
    name: str
    label: str
    violent: bool
    alarms: list[Alarm]
    people: int             # paling banyak orang yang dinilai bersamaan; di bawah 2 berarti tak ada pasangan


def per_class(results: list[PairResult]) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for r in results:
        row = rows.setdefault(r.label, {"n": 0, "violence": 0, "fall": 0, "pair_seen": 0})
        row["n"] += 1
        row["violence"] += any(a.kind == "violence" for a in r.alarms)
        row["fall"] += any(a.kind == "fall" for a in r.alarms)
        row["pair_seen"] += r.people >= 2
    return rows


def evaluate(traces: dict[tuple[str, int], PoseTrace], clips: dict[str, Clip], cfg: Config,
             fall_model: Path | None, violence_model: Path | None) -> list[PairResult]:
    results = []
    for (name, phase), tr in sorted(traces.items()):
        records = replay_log(cfg, tr, fall_model, violence_model, ack_after_s=None)
        alarms = [Alarm(r["t"], r["alarm"]) for r in records if r["alarm"]]
        people = 2 if any(r["pairs"] for r in records) else max((len(r["people"]) for r in records), default=0)
        clip = clips[name]
        results.append(PairResult(f"{name}.p{phase}", clip.label, clip.violent, alarms, min(people, 2)))
    return results


def report(title: str, results: list[PairResult]) -> list[str]:
    rows = per_class(results)
    pct = lambda a, b: f"{100.0 * a / b:5.1f}%" if b else "    - "
    lines = [title, "  aksi        putaran  alarm kekerasan     alarm jatuh (keliru)   pasangan terlihat"]
    total = {True: [0, 0, 0], False: [0, 0, 0]}
    for cls in sorted(CLASSES, key=lambda c: (not CLASSES[c][1], CLASSES[c][0])):
        label, violent = CLASSES[cls]
        row = rows.get(label)
        if row is None:
            continue
        for i, key in enumerate(("n", "violence", "fall")):
            total[violent][i] += row[key]
        lines.append(f"  {label:10s}  {row['n']:5d}    {row['violence']:3d}  {pct(row['violence'], row['n'])}"
                     f"        {row['fall']:3d}  {pct(row['fall'], row['n'])}        {row['pair_seen']:3d}")
    for violent, name in ((True, "KEKERASAN"), (False, "RAMAH")):
        n, v, f = total[violent]
        lines.append(f"  {name:10s}  {n:5d}    {v:3d}  {pct(v, n)}        {f:3d}  {pct(f, n)}")
    return lines


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 0
    root, imgsz = Path(argv[0]), int(argv[1]) if len(argv) > 1 else 416
    cfg = Config()
    clips = {c.name: c for c in load_index(root)}
    traces = build_traces(root, imgsz, cfg.rate_hz)
    out = [f"UT-Interaction lewat YOLO-pose resolusi {imgsz}: {len(clips)} klip x {len(PHASES)} frame awal", ""]
    out += report("Model terpasang, ambang bawaan", evaluate(
        traces, clips, cfg, MODELS_DIR / "fall_model.joblib", MODELS_DIR / "violence_model.joblib"))
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
