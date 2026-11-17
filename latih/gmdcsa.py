"""Pembaca GMDCSA-24: 160 video jatuh dan kegiatan sehari-hari yang direkam dengan webcam laptop.

Sumber: Alam, Sufian, Dutta, Leo, Hameed, "GMDCSA24: A Dataset for Human Fall Detection in Videos",
Data in Brief, 2024. Zenodo doi:10.5281/zenodo.13354453, lisensi CC BY 4.0 (berkas LICENSE di arsip: MIT).

Susunan folder: "Subject N/Fall/NN.mp4", "Subject N/ADL/NN.mp4", dan "Subject N/Fall.csv", "ADL.csv".
Kolom terakhir CSV berisi kelas beserta waktunya, misalnya "Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_CLASS = re.compile(r"([A-Za-z][A-Za-z ()]*?)\s*\[\s*([0-9.]+)\s*to\s*([0-9.]+)\s*\]")


@dataclass(frozen=True)
class Clip:
    name: str               # s1-fall-01
    subject: int
    is_fall: bool
    path: Path
    onset: float | None     # detik awal jatuh; None untuk kegiatan biasa atau bila penyusun tidak memberi waktu
    lying: bool             # kegiatan biasa yang memuat berbaring ("Sleeping")
    partly_hidden: bool     # keterangan penyusun: badan tidak terlihat seluruhnya


def parse_classes(text: str) -> list[tuple[str, float, float]]:
    """[(nama kelas huruf kecil, mulai, selesai)] dari kolom Classes."""
    return [(name.strip().lower(), float(a), float(b)) for name, a, b in _CLASS.findall(text)]


def load_index(root: Path) -> list[Clip]:
    clips = []
    for folder in sorted(Path(root).glob("Subject *")):
        try:
            subject = int(folder.name.split()[-1])
        except ValueError:
            continue
        for kind in ("ADL", "Fall"):
            try:
                lines = (folder / f"{kind}.csv").read_text(encoding="utf-8", errors="replace").splitlines()[1:]
            except OSError:
                continue
            for line in lines:
                file_name = line.split(",", 1)[0].strip()
                path = folder / kind / file_name
                if not file_name.lower().endswith(".mp4") or not path.is_file():
                    continue
                body, _, last = line.rpartition(",")
                classes = parse_classes(last)
                falls = [start for name, start, _ in classes if name.startswith("fall")]
                is_fall = kind == "Fall"
                clips.append(Clip(
                    name=f"s{subject}-{kind.lower()}-{Path(file_name).stem}", subject=subject, is_fall=is_fall, path=path,
                    onset=min(falls) if is_fall and falls else None,
                    lying=not is_fall and any(name.startswith("sleep") for name, _, _ in classes),
                    partly_hidden="not visible" in body.lower()))
    return sorted(clips, key=lambda c: c.name)


# ---- mengukur lewat jalur aplikasi -----------------------------------------------------
#
#     python -m latih.gmdcsa <folder hasil ekstrak> [resolusi]

# Satu frame awal saja: videonya 720p dan panjang, tiga frame awal memakan waktu tiga kali lipat.
PHASES = (0,)


def build_traces(root: Path, imgsz: int, rate_hz: float, phases=PHASES) -> dict:
    """{(nama klip, fase): PoseTrace}. YOLO-pose hanya dijalankan untuk yang belum tersimpan."""
    import time

    from .putar import PoseTrace, trace, video_ticks

    cache = Path(root).parent / f"jejak_{imgsz}"
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
        if fresh:
            print(f"  {n}/{len(clips)} {clip.name}  ({time.monotonic() - started:.0f} dtk)", flush=True)
    return out


def main(argv: list[str]) -> int:
    import sys

    from sahabat.config import MODELS_DIR, Config

    from .urfall import ClipResult, hold, report
    from .putar import replay

    if not argv:
        print(__doc__)
        return 0
    root, imgsz = Path(argv[0]), int(argv[1]) if len(argv) > 1 else 416
    cfg = Config()
    clips = {c.name: c for c in load_index(root)}
    traces = build_traces(root, imgsz, cfg.rate_hz)
    lying = {c.name for c in clips.values() if c.lying}
    model = MODELS_DIR / "fall_model.joblib"
    print(f"GMDCSA-24 lewat YOLO-pose resolusi {imgsz}, {len(traces)} putaran ({len(clips)} klip)\n")
    for title, path in (("Aturan saja", None), ("Model + aturan (bawaan)", model)):
        results = [ClipResult(name, clips[name].is_fall, phase,
                              replay(cfg, tr, fall_model=path, ack_after_s=None), clips[name].onset)
                   for (name, phase), tr in sorted(traces.items())]
        print("\n".join(report(title, results, lying)) + "\n")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
