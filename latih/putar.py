"""Memutar ulang video lewat jalur aplikasi yang asli (YOLO-pose -> riwayat -> penilai -> alarm).

    python -m latih.putar <folder rekaman | berkas video> [--imgsz 416] [--set nama=nilai ...]
                          [--model-jatuh <berkas .joblib>]

Dipakai untuk menyetel ambang tanpa mengulang adegan: rekam sekali lewat "Mulai SAHABAT (rekam uji).bat",
lalu putar ulang berkali-kali dengan ambang berbeda. Hasil YOLO-pose (bagian yang lama) disimpan di
sebelah rekaman sebagai jejak_<resolusi>.npz, jadi putaran berikutnya hanya beberapa detik.

Rekaman disimpan sebagai JPEG, jadi titik tubuh saat diputar ulang bisa sedikit berbeda dari saat langsung.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from sahabat.config import MODELS_DIR, Config
from sahabat.engine import Engine
from sahabat.recorder import read_marks, read_session
from sahabat.types import Person

_EPS = 1e-6
MARK_KIND = {"jatuh": "fall", "kekerasan": "violence"}     # tanda pencatat -> jenis alarm yang seharusnya muncul
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")


# ---- frame mana yang dilihat mesin --------------------------------------------------

def tick_indices(n_frames: int, fps: float, rate_hz: float, phase: int = 0) -> list[tuple[float, int]]:
    """(cap waktu, indeks frame) yang dinilai mesin dari video `fps` bila mesin berputar `rate_hz` kali per detik.

    Mesin mengambil frame terbaru tiap putaran dan hanya menilai frame baru. Kamera yang lebih lambat dari
    mesin memberi tiap frame sekali; `phase` menggeser frame awal (untuk mencoba beberapa keselarasan).
    """
    if fps <= rate_hz + _EPS:
        return [(i / fps, i) for i in range(phase, n_frames)]
    out, k = [], 0
    while True:
        i = int(k * fps / rate_hz + _EPS) + phase
        if i >= n_frames:
            return out
        out.append(((i - phase) / fps, i))
        k += 1


def video_ticks(path: Path, rate_hz: float, phase: int = 0) -> Iterator[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1.0 <= fps <= 240.0 else 25.0
        slow = fps <= rate_hz + _EPS
        i, k = 0, 0                                      # frame ke-i dibaca; putaran ke-k menunggu frame-nya
        while True:
            ok, frame = cap.read()                       # satu per satu: video panjang tidak ditampung di memori
            if not ok or frame is None:
                break
            want = i if slow else int(k * fps / rate_hz + _EPS) + phase
            if i == want and i >= phase:
                yield (i - phase) / fps, frame
                k += 1
            i += 1
    finally:
        cap.release()


def zip_ticks(path: Path, fps: float, rate_hz: float, phase: int = 0) -> Iterator[tuple[float, np.ndarray]]:
    """Klip berupa arsip gambar bernomor (misalnya UR Fall Detection). Arsip tidak diekstrak ke disk."""
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(_IMAGE_EXT))
        for t, i in tick_indices(len(names), fps, rate_hz, phase):
            frame = cv2.imdecode(np.frombuffer(z.read(names[i]), np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield t, frame


def session_ticks(folder: Path) -> Iterator[tuple[float, np.ndarray]]:
    """Rekaman uji: frame dan cap waktu persis seperti yang dinilai mesin saat itu."""
    return read_session(folder)


# ---- jejak titik tubuh ----------------------------------------------------------------

@dataclass
class PoseTrace:
    """Hasil YOLO-pose untuk tiap frame yang dinilai. Cukup untuk mengulang penilaian tanpa videonya."""
    t: np.ndarray
    frame_h: int
    frame_w: int
    people: list[list[Person]]

    def save(self, path: Path) -> None:
        flat = [p for ps in self.people for p in ps]
        np.savez_compressed(
            path, t=np.asarray(self.t, dtype=np.float64), shape=np.array([self.frame_h, self.frame_w]),
            counts=np.array([len(ps) for ps in self.people], dtype=np.int32),
            ids=np.array([p.track_id for p in flat], dtype=np.int64),
            kp=np.array([p.kp for p in flat], dtype=np.float32).reshape(-1, 17, 2),
            conf=np.array([p.conf for p in flat], dtype=np.float32).reshape(-1, 17),
            box=np.array([p.box for p in flat], dtype=np.float32).reshape(-1, 4))

    @classmethod
    def load(cls, path: Path) -> "PoseTrace":
        with np.load(path, allow_pickle=False) as z:
            t, shape, counts = z["t"], z["shape"], z["counts"]
            ids, kp, conf, box = z["ids"], z["kp"], z["conf"], z["box"]
        people, at = [], 0
        for n in counts:
            people.append([Person(int(ids[j]), kp[j], conf[j], tuple(float(v) for v in box[j]))
                           for j in range(at, at + int(n))])
            at += int(n)
        return cls(t, int(shape[0]), int(shape[1]), people)


def trace(frames: Iterable[tuple[float, np.ndarray]], pose) -> PoseTrace:
    ts, people, shape = [], [], (480, 640)
    for t, frame in frames:
        ts.append(t)
        people.append(pose.detect(frame))
        shape = frame.shape[:2]
    return PoseTrace(np.array(ts, dtype=np.float64), int(shape[0]), int(shape[1]), people)


# ---- mengulang penilaian --------------------------------------------------------------

@dataclass(frozen=True)
class Alarm:
    t: float
    kind: str               # fall | violence


class _TracePose:
    def __init__(self, people: list[list[Person]]) -> None:
        self._people, self._at = people, 0

    def detect(self, frame) -> list[Person]:
        got = self._people[self._at]
        self._at += 1
        return got


class _Collect:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)

    def close(self) -> None:
        pass


def replay_log(cfg: Config, tr: PoseTrace, fall_model: Path | None = None, violence_model: Path | None = None,
               ack_after_s: float | None = 3.0) -> list[dict]:
    """Catatan per putaran (sama dengan log prediksi aplikasi) bila jejak ini dinilai dengan ambang `cfg`.

    `ack_after_s`: "Sudah diperiksa" ditekan sekian detik setelah tiap alarm (None = tidak pernah).
    """
    log = _Collect()
    engine = Engine(cfg, None, _TracePose(tr.people), log, violence_model=violence_model, fall_model=fall_model)
    blank = np.zeros((tr.frame_h, tr.frame_w, 3), dtype=np.uint8)
    red_since = None
    for t in tr.t:
        snap = engine.step(float(t), blank)
        if snap.bar in ("jatuh", "kekerasan"):
            red_since = float(t) if red_since is None else red_since
            if ack_after_s is not None and t - red_since >= ack_after_s - _EPS:
                engine.acknowledge()
                red_since = None
        else:
            red_since = None
    return log.records


def replay(cfg: Config, tr: PoseTrace, fall_model: Path | None = None, violence_model: Path | None = None,
           ack_after_s: float | None = 3.0) -> list[Alarm]:
    """Alarm yang akan muncul bila jejak ini dinilai dengan ambang `cfg`, lewat mesin yang sama dengan aplikasi."""
    return [Alarm(r["t"], r["alarm"]) for r in replay_log(cfg, tr, fall_model, violence_model, ack_after_s)
            if r["alarm"]]


# ---- mencocokkan dengan tanda pencatat -----------------------------------------------

@dataclass(frozen=True)
class MarkRow:
    t: float
    label: str
    ok: bool                 # jatuh/kekerasan: alarm yang benar muncul; biasa: tidak ada alarm
    delay_s: float | None    # waktu alarm yang benar relatif ke tanda (negatif = sebelum tombol ditekan)
    other: list[str]         # alarm lain di rentang yang sama


def match_marks(alarms: list[Alarm], marks: list[tuple[float, str]],
                before_s: float = 10.0, after_s: float = 3.0) -> list[MarkRow]:
    """Tanda ditekan pencatat setelah adegan selesai, jadi alarmnya dicari sampai `before_s` sebelum tanda."""
    rows, prev = [], float("-inf")
    for t, label in sorted(marks):
        near = [a for a in alarms if max(t - before_s, prev) < a.t <= t + after_s]
        want = MARK_KIND.get(label)
        right = [a for a in near if a.kind == want]
        if want is None:
            rows.append(MarkRow(t, label, not near, None, [a.kind for a in near]))
        else:
            rows.append(MarkRow(t, label, bool(right), right[0].t - t if right else None,
                                [a.kind for a in near if a.kind != want]))
        prev = t
    return rows


# ---- baris perintah ---------------------------------------------------------------------

def load_config_snapshot(path: Path) -> Config:
    """Muat konfigurasi eksperimen, dengan dukungan snapshot dari revisi proyek lain."""
    if not path.is_file():
        return Config()
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Snapshot konfigurasi harus berupa objek JSON")
    supported = {field.name for field in dataclasses.fields(Config)}
    return Config(**{name: value for name, value in values.items() if name in supported})


def _overrides(args: list[str]) -> dict:
    fields = {f.name: f for f in dataclasses.fields(Config)}
    out = {}
    for item in args:
        name, _, raw = item.partition("=")
        if name not in fields:
            raise SystemExit(f"Tidak ada ambang bernama {name!r} di sahabat/config.py")
        current = getattr(Config(), name)
        if name == "fall_rule_model_threshold":
            out[name] = None if raw.lower() == "none" else float(raw)
        else:
            out[name] = type(current)(float(raw)) if isinstance(current, (int, float)) else raw
    return out


def load_or_trace(path: Path, imgsz: int, rate_hz: float) -> PoseTrace:
    """Jejak dari berkas simpanan bila ada; kalau belum, jalankan YOLO-pose lalu simpan."""
    path = Path(path)
    cache = (path / f"jejak_{imgsz}.npz") if path.is_dir() else path.with_name(f"{path.stem}.jejak_{imgsz}.npz")
    if cache.is_file():
        return PoseTrace.load(cache)
    from sahabat.pose import YoloPose
    frames = session_ticks(path) if path.is_dir() else video_ticks(path, rate_hz)
    tr = trace(frames, YoloPose(imgsz))
    try:
        tr.save(cache)
    except OSError:
        pass
    return tr


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    path, imgsz, sets = Path(argv[0]), 416, []
    fall_model = MODELS_DIR / "fall_model.joblib"
    rest = argv[1:]
    while rest:
        flag = rest.pop(0)
        if flag == "--imgsz" and rest:
            imgsz = int(rest.pop(0))
        elif flag == "--model-jatuh" and rest:
            fall_model = Path(rest.pop(0))               # misalnya models\fall_model.ntu-saja.joblib
            if not fall_model.is_file():
                raise SystemExit(f"Model tidak ditemukan: {fall_model}")
        elif flag == "--set":
            while rest and not rest[0].startswith("--"):
                sets.append(rest.pop(0))
        else:
            raise SystemExit(f"Argumen tidak dikenal: {flag}")
    cfg = dataclasses.replace(Config(), **_overrides(sets))
    tr = load_or_trace(path, imgsz, cfg.rate_hz)
    if len(tr.t) == 0:
        print("Tidak ada frame yang terbaca.")
        return 1
    alarms = replay(cfg, tr, fall_model, MODELS_DIR / "violence_model.joblib")
    t0, span = float(tr.t[0]), float(tr.t[-1] - tr.t[0])
    print(f"{len(tr.t)} frame, {span:.1f} detik, {len(tr.t) / span if span > 0 else 0:.1f} frame per detik")
    print(f"{len(alarms)} alarm:")
    for a in alarms:
        print(f"  {a.t - t0:8.1f} dtk  {a.kind}")
    marks = read_marks(path) if path.is_dir() else []
    if marks:
        rows = match_marks(alarms, marks)
        print(f"{len(rows)} tanda pencatat:")
        for r in rows:
            delay = "" if r.delay_s is None else f"  alarm {r.delay_s:+.1f} dtk dari tanda"
            other = f"  (juga: {', '.join(r.other)})" if r.other else ""
            print(f"  {r.t - t0:8.1f} dtk  {r.label:10s} {'BENAR' if r.ok else 'SALAH'}{delay}{other}")
        for label in sorted({r.label for r in rows}):
            mine = [r for r in rows if r.label == label]
            print(f"  {label}: {sum(r.ok for r in mine)} dari {len(mine)} benar")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
