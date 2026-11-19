"""Riwayat kejadian dan klip bukti.

Setiap kejadian (jatuh / indikasi kekerasan) dicatat di data/riwayat/riwayat.jsonl, termasuk kejadian yang
muncul saat alarm lain masih menyala. Bila penyimpanan klip hidup, frame layar (dengan kerangka) dari
`before_s` detik sebelum sampai `after_s` detik sesudah kejadian disimpan sebagai satu zip berisi JPEG di
data/riwayat/klip/. Catatan dan klip dihapus otomatis setelah `keep_days` hari. Semua hanya di laptop ini.
"""
from __future__ import annotations

import io
import json
import shutil
import threading
import time
import zipfile
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

LOG_FILE = "riwayat.jsonl"
CLIP_DIR = "klip"
TIMES_FILE = "waktu.txt"
MIN_FREE_BYTES = 1024 ** 3          # klip tidak disimpan bila disk hampir penuh; catatan tetap ditulis
MAX_LISTED = 200


def _free_bytes(folder: Path) -> int:
    return shutil.disk_usage(folder).free


@dataclass
class Kejadian:
    id: str
    waktu: float                    # detik sejak epoch (jam laptop)
    jenis: str                      # "jatuh" | "kekerasan"
    mode: str                       # "full" | "upper" | "none"
    orang: list[int]
    sumber: str
    klip: bool
    kamera: str = ""                # Mode CCTV: ID kamera; "" = kamera Mode Pameran
    kamera_nama: str = ""


@dataclass
class _Capture:
    record: Kejadian
    end_t: float
    frames: list[tuple[float, bytes]]
    kamera: str = ""


class Riwayat:
    def __init__(self, folder: Path, *, keep_clips: bool = True, before_s: float = 5.0, after_s: float = 3.0,
                 keep_days: float = 7.0, clock: Callable[[], float] = time.time,
                 free_bytes: Callable[[Path], int] = _free_bytes, min_free: int = MIN_FREE_BYTES,
                 background: bool = True) -> None:
        self.folder = Path(folder)
        self.keep_clips = keep_clips
        self.before_s = before_s
        self.after_s = after_s
        self.keep_days = keep_days
        self._clock = clock
        self._free_bytes = free_bytes
        self.min_free = min_free
        self._background = background
        self._lock = threading.RLock()
        self._rings: dict[str, deque[tuple[float, bytes]]] = {}     # per kamera ("" = Mode Pameran)
        self._open: list[_Capture] = []
        self._writers: list[threading.Thread] = []
        self._used_ids: set[str] = set()
        self.seq = 0                 # naik tiap kejadian baru; layar memakainya untuk angka di tombol Riwayat
        self.disabled = False
        try:
            (self.folder / CLIP_DIR).mkdir(parents=True, exist_ok=True)
        except OSError:              # folder hanya-baca: aplikasi tetap jalan tanpa riwayat
            self.disabled = True
            print("Peringatan: riwayat kejadian tidak bisa disimpan; lanjut tanpa riwayat.")
            return
        self.prune()
        self.seq = len(self._read_all())

    # ---- masukan dari mesin -------------------------------------------------------

    @property
    def wants_frames(self) -> bool:
        return self.keep_clips and not self.disabled

    def set_keep_clips(self, on: bool) -> bool:
        with self._lock:
            self.keep_clips = bool(on)
            if not on:
                self._rings.clear()
            return self.keep_clips

    def frame(self, t: float, jpeg: bytes, kamera: str = "") -> None:
        """Frame layar terbaru (JPEG, sudah berkerangka) satu kamera. Dipanggil tiap putaran penilaian."""
        if not self.wants_frames:
            return
        with self._lock:
            ring = self._rings.setdefault(kamera, deque())
            ring.append((t, jpeg))
            while ring and t - ring[0][0] > self.before_s + 0.5:
                ring.popleft()
            done = []
            for cap in self._open:
                if cap.kamera != kamera:
                    continue
                if t <= cap.end_t + 1e-6:
                    cap.frames.append((t, jpeg))
                if t >= cap.end_t - 1e-6:
                    done.append(cap)
            for cap in done:
                self._open.remove(cap)
                self._finish(cap)

    def event(self, t: float, jenis: str, mode: str, ids, sumber: str = "model", kamera: str = "",
              kamera_nama: str = "") -> Kejadian | None:
        """Kejadian baru pada waktu mesin t. Klip mulai dikumpulkan bila penyimpanan klip hidup."""
        if self.disabled:
            return None
        with self._lock:
            clip = self.wants_frames and self._has_space()
            record = Kejadian(self._new_id(), round(self._clock(), 3), jenis, mode, [int(i) for i in ids],
                              sumber, clip, kamera, kamera_nama)
            self._append(record)
            self.seq += 1
            if clip:
                ring = self._rings.get(kamera, ())
                frames = [(ft, j) for ft, j in ring if ft >= t - self.before_s - 1e-6]
                self._open.append(_Capture(record, t + self.after_s, frames, kamera))
            return record

    def close(self) -> None:
        """Aplikasi ditutup: klip yang belum lengkap disimpan sejauh yang ada."""
        with self._lock:
            pending, self._open = self._open, []
            for cap in pending:
                self._finish(cap)
            writers = list(self._writers)
        for w in writers:
            w.join(timeout=10.0)

    # ---- dibaca layar --------------------------------------------------------------

    def list(self, limit: int = MAX_LISTED, kamera: str | None = None) -> list[dict]:
        """Kejadian terbaru lebih dulu, bisa disaring per kamera. `klip` hanya benar bila berkasnya sudah tersimpan
        dan masih ada."""
        if self.disabled:
            return []
        with self._lock:
            open_ids = {c.record.id for c in self._open}
            out = []
            records = [r for r in self._read_all() if kamera is None or r.kamera == kamera]
            for rec in reversed(records[-limit:]):
                item = asdict(rec)
                item["klip"] = rec.klip and rec.id not in open_ids and self._clip_path(rec.id).is_file()
                item["menyimpan"] = rec.id in open_ids
                out.append(item)
            return out

    def clip(self, record_id: str) -> list[tuple[float, bytes]] | None:
        """Frame klip (detik sejak frame pertama, JPEG). None bila tidak ada."""
        if self.disabled or not _safe_id(record_id):
            return None
        path = self._clip_path(record_id)
        try:
            with zipfile.ZipFile(path) as z:
                times = [float(x) for x in z.read(TIMES_FILE).decode("ascii").split()]
                names = sorted(n for n in z.namelist() if n.endswith(".jpg"))
                return [(t, z.read(n)) for t, n in zip(times, names)]
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            return None

    # ---- perawatan -------------------------------------------------------------------

    def prune(self) -> None:
        """Menghapus catatan dan klip yang lebih tua dari keep_days hari, serta klip tanpa catatan."""
        if self.disabled:
            return
        with self._lock:
            cutoff = self._clock() - self.keep_days * 86400
            records = self._read_all()
            keep = [r for r in records if r.waktu >= cutoff]
            if len(keep) != len(records):
                self._rewrite(keep)
            ids = {r.id for r in keep}
            for path in (self.folder / CLIP_DIR).glob("*.zip"):
                if path.stem not in ids:
                    try:
                        path.unlink()
                    except OSError:
                        pass

    # ---- bagian dalam ---------------------------------------------------------------

    def _has_space(self) -> bool:
        try:
            return self._free_bytes(self.folder) >= self.min_free
        except OSError:
            return False

    def _new_id(self) -> str:
        base = datetime.fromtimestamp(self._clock()).strftime("%Y%m%d-%H%M%S")
        n = 1
        while f"{base}-{n}" in self._used_ids or self._clip_path(f"{base}-{n}").exists():
            n += 1
        self._used_ids.add(f"{base}-{n}")
        return f"{base}-{n}"

    def _clip_path(self, record_id: str) -> Path:
        return self.folder / CLIP_DIR / f"{record_id}.zip"

    def _log_path(self) -> Path:
        return self.folder / LOG_FILE

    def _append(self, record: Kejadian) -> None:
        try:
            with open(self._log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _rewrite(self, records: list[Kejadian]) -> None:
        tmp = self._log_path().with_suffix(".tmp")
        try:
            tmp.write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in records),
                           encoding="utf-8")
            tmp.replace(self._log_path())
        except OSError:
            pass

    def _read_all(self) -> list[Kejadian]:
        try:
            lines = self._log_path().read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                raw = json.loads(line)
                out.append(Kejadian(str(raw["id"]), float(raw["waktu"]), str(raw["jenis"]), str(raw["mode"]),
                                    [int(i) for i in raw["orang"]], str(raw.get("sumber", "model")),
                                    bool(raw.get("klip", False)), str(raw.get("kamera", "")),
                                    str(raw.get("kamera_nama", ""))))
            except (ValueError, KeyError, TypeError):
                continue
        for r in out:
            self._used_ids.add(r.id)
        return out

    def _finish(self, cap: _Capture) -> None:
        if not cap.frames:
            return
        if self._background:
            w = threading.Thread(target=self._write_clip, args=(cap,), name="klip", daemon=True)
            self._writers = [x for x in self._writers if x.is_alive()] + [w]
            w.start()
        else:
            self._write_clip(cap)

    def _write_clip(self, cap: _Capture) -> None:
        t0 = cap.frames[0][0]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:          # JPEG sudah termampatkan
            z.writestr(TIMES_FILE, "\n".join(f"{t - t0:.3f}" for t, _ in cap.frames))
            for i, (_, jpeg) in enumerate(cap.frames):
                z.writestr(f"{i:05d}.jpg", jpeg)
        path = self._clip_path(cap.record.id)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_bytes(buf.getvalue())
            tmp.replace(path)
        except OSError:
            pass


def _safe_id(record_id: str) -> bool:
    """Nomor kejadian hanya angka dan tanda hubung, supaya tidak bisa menunjuk berkas lain."""
    return bool(record_id) and len(record_id) <= 40 and all(c.isdigit() or c == "-" for c in record_id)
