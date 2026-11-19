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


def app():
    engine = StubEngine()
    listener = StubListener()
    # panel "HP / laptop lain" ikut tampil; server keduanya hanya di 127.0.0.1 supaya tidak memicu firewall
    server.remember_lan = lambda on, code: None             # jangan menulis ke pengaturan laptop sungguhan
    lan = LanServer(0, "ABCD2345", host="127.0.0.1", addresses=lambda: ["192.168.1.13"])
    import tempfile
    from pathlib import Path

    from sahabat.kamera import KameraStore
    kamera = KameraStore(Path(tempfile.mkdtemp(prefix="sahabat-stub-")) / "kamera.json")   # bukan data/ sungguhan
    application = create_app(engine, listener, lan=lan, kamera=kamera)
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
