"""Menguji deteksi jatuh pada video sungguhan (UR Fall Detection) lewat jalur aplikasi yang asli.

    python -m latih.urfall <folder berisi fall-NN-cam0-rgb.zip dan adl-NN-cam0-rgb.zip> [resolusi]

Berbeda dari latih.fall_train (titik tubuh HRNet dari kamera Kinect NTU), di sini gambar RGB dilewatkan
YOLO-pose + ByteTrack yang sama dengan aplikasi, lalu mesin yang sama, 10 kali per detik. Tiap klip dinilai
tiga kali dengan frame awal berbeda (video 30 gambar per detik, mesin hanya melihat tiap frame ketiga).

Sumber: Kwolek & Kepski, "Human fall detection on embedded platform using depth maps and wireless
accelerometer", CMPB 117(3), 2014. Lisensi CC BY-NC-SA 4.0. 30 klip jatuh + 40 klip kegiatan sehari-hari,
kamera 0 (sejajar lantai). Sejak 14 November 2026 klipnya juga ikut melatih penilai jatuh (latih.fall_train),
dengan validasi silang per klip; karena lisensinya non-komersial, model hasilnya juga hanya untuk itu.
"""
from __future__ import annotations

import csv
import dataclasses
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sahabat.config import MODELS_DIR, Config

from .putar import Alarm, PoseTrace, replay, trace, zip_ticks

FPS = 30.0
HOLD_S = 1.5              # klip berakhir tak lama setelah orangnya rebah; pose terakhir ditahan seperti di NTU
PHASES = (0, 1, 2)
LABEL_FALLING, LABEL_LYING = 0, 1


def _rows(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                try:
                    yield row[0].strip(), int(row[1]), int(float(row[2]))
                except ValueError:
                    continue


def fall_onsets(csv_path: Path, fps: float = FPS) -> dict[str, float]:
    """Detik awal jatuh per klip: frame pertama berlabel "sedang jatuh" (atau "tergeletak" bila tak ada)."""
    first: dict[str, dict[int, int]] = {}
    for name, frame, label in _rows(csv_path):
        if label in (LABEL_FALLING, LABEL_LYING):
            seen = first.setdefault(name, {})
            seen[label] = min(seen.get(label, frame), frame)
    return {name: (seen.get(LABEL_FALLING, seen.get(LABEL_LYING)) - 1) / fps
            for name, seen in first.items() if name.startswith("fall")}


def fall_times(csv_path: Path, fps: float = FPS) -> dict[str, tuple[float, float | None]]:
    """(detik awal jatuh, detik mulai tergeletak) per klip jatuh; yang kedua None bila tak ada labelnya."""
    first: dict[str, dict[int, int]] = {}
    for name, frame, label in _rows(csv_path):
        if name.startswith("fall") and label in (LABEL_FALLING, LABEL_LYING):
            seen = first.setdefault(name, {})
            seen[label] = min(seen.get(label, frame), frame)
    out = {}
    for name, seen in first.items():
        lying = (seen[LABEL_LYING] - 1) / fps if LABEL_LYING in seen else None
        out[name] = ((seen.get(LABEL_FALLING, seen.get(LABEL_LYING)) - 1) / fps, lying)
    return out


def lying_clips(csv_path: Path) -> set[str]:
    """Klip yang orangnya sampai tergeletak (pada kegiatan sehari-hari: berbaring dengan sengaja)."""
    return {name for name, _, label in _rows(csv_path) if label == LABEL_LYING}


def hold(tr: PoseTrace, seconds: float, rate_hz: float) -> PoseTrace:
    """Jejak dengan pose terakhir diulang `seconds` detik, seolah orangnya diam setelah klip habis."""
    n = int(round(seconds * rate_hz))
    if n <= 0 or len(tr.t) == 0:
        return tr
    extra = float(tr.t[-1]) + np.arange(1, n + 1) / rate_hz
    return PoseTrace(np.concatenate([tr.t, extra]), tr.frame_h, tr.frame_w, tr.people + [tr.people[-1]] * n)


@dataclass
class ClipResult:
    name: str                 # fall-01, adl-17, ...
    is_fall: bool
    phase: int
    alarms: list[Alarm]
    onset: float | None       # detik awal jatuh; None untuk kegiatan sehari-hari


@dataclass
class Summary:
    falls: int = 0
    falls_caught: int = 0
    delays: list[float] = field(default_factory=list)          # alarm pertama setelah awal jatuh
    early: list[str] = field(default_factory=list)             # klip jatuh yang alarmnya muncul sebelum jatuh
    adls: int = 0
    adl_false: int = 0
    false_names: list[str] = field(default_factory=list)
    missed_names: list[str] = field(default_factory=list)
    violence_names: list[str] = field(default_factory=list)


def _add(names: list[str], name: str) -> None:
    if name not in names:
        names.append(name)


def summarize(results: list[ClipResult]) -> Summary:
    s = Summary()
    for r in results:
        falls = [a.t for a in r.alarms if a.kind == "fall"]
        if any(a.kind == "violence" for a in r.alarms):
            _add(s.violence_names, r.name)
        if r.is_fall:
            s.falls += 1
            onset = r.onset or 0.0
            after = [t for t in falls if t >= onset]
            if any(t < onset for t in falls):
                _add(s.early, r.name)
            if after:
                s.falls_caught += 1
                s.delays.append(after[0] - onset)
            else:
                _add(s.missed_names, r.name)
        else:
            s.adls += 1
            if falls:
                s.adl_false += 1
                _add(s.false_names, r.name)
    return s


def build_traces(folder: Path, imgsz: int, rate_hz: float) -> dict[tuple[str, int], PoseTrace]:
    """YOLO-pose untuk semua klip dan fase; hasilnya disimpan supaya menyetel ambang tidak mengulang inferensi."""
    cache = folder / f"jejak_{imgsz}"
    cache.mkdir(exist_ok=True)
    pose, out, started = None, {}, time.monotonic()
    zips = sorted(folder.glob("*-cam0-rgb.zip"))
    for n, path in enumerate(zips, 1):
        name = path.name.replace("-cam0-rgb.zip", "")
        fresh = False
        for phase in PHASES:
            saved = cache / f"{name}.p{phase}.npz"
            if saved.is_file():
                out[name, phase] = PoseTrace.load(saved)
                continue
            if pose is None:
                from sahabat.pose import YoloPose
                pose = YoloPose(imgsz)
            pose.reset()                                 # klip baru: nomor pelacak mulai dari awal
            tr = trace(zip_ticks(path, FPS, rate_hz, phase), pose)
            tr.save(saved)
            out[name, phase] = tr
            fresh = True
        if fresh:
            print(f"  {n}/{len(zips)} {name}  ({time.monotonic() - started:.0f} dtk)", flush=True)
    return out


def evaluate(traces: dict[tuple[str, int], PoseTrace], onsets: dict[str, float], cfg: Config,
             fall_model: Path | None, hold_s: float) -> list[ClipResult]:
    results = []
    for (name, phase), tr in sorted(traces.items()):
        alarms = replay(cfg, hold(tr, hold_s, cfg.rate_hz), fall_model=fall_model, ack_after_s=None)
        results.append(ClipResult(name, name.startswith("fall"), phase, alarms, onsets.get(name)))
    return results


def report(title: str, results: list[ClipResult], lying: set[str]) -> list[str]:
    s = summarize(results)
    lie = summarize([r for r in results if not r.is_fall and r.name in lying])
    rest = summarize([r for r in results if not r.is_fall and r.name not in lying])
    d = sorted(s.delays)
    pct = lambda a, b: f"{100.0 * a / b:5.1f}%" if b else "   - "
    lines = [title,
             f"  jatuh beralarm          : {s.falls_caught:3d} dari {s.falls:3d}  {pct(s.falls_caught, s.falls)}",
             f"  dalam 3 detik           : {sum(x <= 3.0 for x in d):3d} dari {s.falls:3d}  "
             f"{pct(sum(x <= 3.0 for x in d), s.falls)}",
             f"  waktu peringatan        : median {np.median(d):.1f} dtk, terlama {d[-1]:.1f} dtk" if d else
             "  waktu peringatan        : -",
             f"  kegiatan biasa beralarm : {s.adl_false:3d} dari {s.adls:3d}  {pct(s.adl_false, s.adls)}",
             f"    berbaring sengaja     : {lie.adl_false:3d} dari {lie.adls:3d}  {pct(lie.adl_false, lie.adls)}",
             f"    lainnya               : {rest.adl_false:3d} dari {rest.adls:3d}  {pct(rest.adl_false, rest.adls)}"]
    if s.missed_names:
        lines.append(f"  jatuh terlewat          : {', '.join(s.missed_names)}")
    if s.false_names:
        lines.append(f"  alarm keliru            : {', '.join(s.false_names)}")
    if s.early:
        lines.append(f"  alarm sebelum jatuh     : {', '.join(s.early)}")
    if s.violence_names:
        lines.append(f"  alarm kekerasan (keliru): {', '.join(s.violence_names)}")
    return lines


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 0
    folder = Path(argv[0])
    imgsz = int(argv[1]) if len(argv) > 1 else 416
    cfg = Config()
    onsets = fall_onsets(folder / "urfall-cam0-falls.csv")
    lying = lying_clips(folder / "urfall-cam0-adls.csv")
    print(f"YOLO-pose resolusi {imgsz}:", flush=True)
    traces = build_traces(folder, imgsz, cfg.rate_hz)
    model = MODELS_DIR / "fall_model.joblib"
    out = [f"UR Fall Detection lewat YOLO-pose resolusi {imgsz}, {len(traces)} putaran "
           f"({len({n for n, _ in traces})} klip x {len(PHASES)} frame awal)", ""]
    for title, path, hold_s in (("Aturan saja, pose terakhir ditahan 1,5 dtk", None, HOLD_S),
                                ("Model + aturan (bawaan), pose terakhir ditahan 1,5 dtk", model, HOLD_S),
                                ("Model + aturan (bawaan), klip apa adanya", model, 0.0)):
        out += report(title, evaluate(traces, onsets, cfg, path, hold_s), lying) + [""]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
