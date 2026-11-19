"""Server tiruan untuk memeriksa tampilan tanpa kamera dan tanpa model.

    runtime/python/python.exe -m uvicorn --factory tests.stub_server:app --port 8766
    buka http://localhost:8766 lalu ganti keadaan lewat /stub/jatuh, /stub/kekerasan, /stub/aman, ...
    (tambahkan ?rekam=true untuk menampilkan tanda MEREKAM)
"""
from __future__ import annotations

import cv2
import numpy as np

from sahabat.engine import BAR_TEXT, Snapshot
from sahabat.jaringan import LanServer
from sahabat.listener import NOTICE_TEXT, STATUS_LABEL, ListenState
from sahabat import server
from sahabat.server import create_app


class StubEngine:
    def __init__(self) -> None:
        self.bar, self.seq, self.slow = "aman", 0, False
        self.recording, self.marks = False, []

    def snapshot(self) -> Snapshot:
        return Snapshot(self.bar, BAR_TEXT[self.bar], 10.0, 2, self.slow, self.seq, (), "", self.recording)

    def mark(self, label: str) -> bool:
        self.marks.append(label)
        return self.recording

    def acknowledge(self) -> bool:
        was_red = self.bar in ("jatuh", "kekerasan")
        if was_red:
            self.bar = "aman"
        return was_red

    def latest_jpeg(self) -> bytes:
        canvas = np.full((480, 640, 3), 60, dtype=np.uint8)
        cv2.putText(canvas, "video tiruan", (190, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
        return cv2.imencode(".jpg", canvas)[1].tobytes()


class StubListener:
    def __init__(self) -> None:
        self.status, self.notice, self.seq, self.calls = "hidup", "", 0, []

    def snapshot(self) -> ListenState:
        return ListenState(self.status, STATUS_LABEL[self.status], self.notice, self.seq)

    def set_enabled(self, on: bool) -> bool:
        self.calls.append(on)
        self.status = "hidup" if on else "mati"
        return on


class StubCctv:
    """Mode CCTV tiruan: 3 kamera bergambar warna; bar per kamera disetel lewat /stub/cctv/<kamera>/<bar>."""

    COLORS = {"k1": (40, 90, 40), "k2": (40, 40, 110), "k3": (90, 60, 30), "k4": (70, 30, 70), "k5": (30, 80, 90),
              "k6": (90, 90, 30), "k7": (50, 50, 50)}

    def __init__(self, store) -> None:
        import time
        self.store, self.active, self.slow = store, False, False
        self.bars = {k: "aman" for k in self.COLORS}
        self.since: dict[str, float] = {}
        self.seq = 0
        self._time = time

    def start(self):
        self.active = True
        return self.snapshot()

    def stop(self):
        self.active = False

    def reload(self):
        pass

    def set_bar(self, cam: str, bar: str) -> None:
        if bar in ("jatuh", "kekerasan") and self.bars.get(cam) not in ("jatuh", "kekerasan"):
            self.seq += 1
            self.since[cam] = self._time.time()
        self.bars[cam] = bar

    def snapshot(self) -> dict:
        names = {c["id"]: c["nama"] for c in self.store.list_public()}
        zones = {c["id"]: c["zona"] for c in self.store.list_public()}
        red = sorted((k for k, b in self.bars.items() if b in ("jatuh", "kekerasan")), key=lambda k: self.since[k])
        rows = []
        for k, b in self.bars.items():
            status = "terputus" if b == "kamera_terputus" else ("lambat" if self.slow and k == "k3" else
                                                                ("penuh" if b != "aman" or k == "k1" else "siaga"))
            rows.append({"id": k, "nama": names.get(k, k), "bar": b, "text": BAR_TEXT[b], "fps": 10.0, "people": 1,
                         "status": status, "prioritas": k == "k1", "zona": zones.get(k), "galat": ""})
        if red:
            first = red[0]
            stamp = self._time.strftime("%H.%M.%S", self._time.localtime(self.since[first]))
            text = f"{BAR_TEXT[self.bars[first]]} — {names.get(first, first)} — {stamp}"
            if self.slow and first == "k3":
                text += " (dinilai lambat)"
            bar = self.bars[first]
        else:
            bar, text = "aman", BAR_TEXT["aman"]
        return {"aktif": self.active, "bar": bar, "text": text, "fps": 10.0, "people": 3, "alarm_seq": self.seq,
                "alarm_kamera": red[0] if red else None, "alarm_list": red, "kapasitas": 1, "lambat": self.slow,
                "kamera": rows}

    def has_unacked(self, cam=None):
        return any(b in ("jatuh", "kekerasan") for k, b in self.bars.items() if cam in (None, k))

    def acknowledge(self, cam=None):
        cam = cam or self.snapshot()["alarm_kamera"]
        if cam and self.bars.get(cam) in ("jatuh", "kekerasan"):
            self.bars[cam] = "aman"
            return True
        return False

    def acknowledge_all(self):
        return sum(1 for k in list(self.bars) if self.acknowledge(k))

    def set_zone(self, cam, points):
        from sahabat import zone as zones
        return zones.normalize(points)

    def latest_jpeg(self, cam, small=False):
        name = {c["id"]: c["nama"] for c in self.store.list_public()}.get(cam, cam)
        h, w = (270, 480) if small else (480, 640)
        canvas = np.full((h, w, 3), self.COLORS.get(cam, (60, 60, 60)), dtype=np.uint8)
        cv2.putText(canvas, name, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
        return cv2.imencode(".jpg", canvas)[1].tobytes()


def app():
    engine = StubEngine()
    engine.pause = engine.resume = lambda: None
    listener = StubListener()
    # panel "HP / laptop lain" ikut tampil; server keduanya hanya di 127.0.0.1 supaya tidak memicu firewall
    server.remember_lan = lambda on, code: None             # jangan menulis ke pengaturan laptop sungguhan
    lan = LanServer(0, "ABCD2345", host="127.0.0.1", addresses=lambda: ["192.168.1.13"])
    import tempfile
    from pathlib import Path

    from sahabat.kamera import KameraStore
    import json
    folder = Path(tempfile.mkdtemp(prefix="sahabat-stub-"))                                # bukan data/ sungguhan
    (folder / "kamera.json").write_text(json.dumps({"kamera": [
        {"id": i, "nama": n, "jenis": "alamat", "alamat": f"rtsp://192.168.1.{60 + j}:554/stream2", "pengguna": "",
         "sandi": "", "aktif": True, "prioritas": i == "k1", "zona": None, "potong": None}
        for j, (i, n) in enumerate([("k1", "Lorong A"), ("k2", "Kelas 7B"), ("k3", "Kantin"), ("k4", "Tangga"),
                                    ("k5", "Lapangan"), ("k6", "Gerbang"), ("k7", "Perpustakaan")])]}),
        encoding="utf-8")
    kamera = KameraStore(folder / "kamera.json")
    from sahabat.cctv import ModeControl
    cctv = StubCctv(kamera)
    mode = ModeControl(engine, lambda: cctv, remember=lambda on: None)
    application = create_app(engine, listener, lan=lan, kamera=kamera, mode=mode)

    @application.get("/stub/cctv/{cam}/{bar}")
    async def set_cctv_bar(cam: str, bar: str):
        """/stub/cctv/k2/jatuh -> alarm di kamera k2; /stub/cctv/k3/kamera_terputus -> ubin terputus."""
        if bar not in BAR_TEXT or cam not in cctv.bars:
            return {"ok": False, "pilihan": list(BAR_TEXT), "kamera": list(cctv.bars)}
        cctv.set_bar(cam, bar)
        return {"ok": True}

    @application.get("/stub/cctv-lambat")
    async def set_cctv_slow(on: int = 1):
        cctv.slow = bool(on)
        return {"ok": True, "lambat": cctv.slow}
    lan.app = application

    @application.get("/stub-kata")
    async def set_words(status: str = "hidup", teks: str = ""):
        """/stub-kata?status=hidup&teks=a***r  -> pita kuning; teks kosong -> pita hilang."""
        if status not in STATUS_LABEL:
            return {"ok": False, "pilihan": list(STATUS_LABEL)}
        listener.status = status
        listener.notice = NOTICE_TEXT.format(teks) if teks else ""
        if teks:
            listener.seq += 1
        return {"ok": True, "status": status, "tekanan_tombol": listener.calls}

    @application.get("/stub/{bar}")
    async def set_bar(bar: str, slow: bool = False, rekam: bool | None = None):
        if rekam is not None:
            engine.recording = rekam
        if bar not in BAR_TEXT:
            return {"ok": False, "pilihan": list(BAR_TEXT)}
        if bar in ("jatuh", "kekerasan"):
            engine.seq += 1
        engine.bar, engine.slow = bar, slow
        return {"ok": True, "bar": bar, "rekam": engine.recording, "tanda": engine.marks}

    return application
