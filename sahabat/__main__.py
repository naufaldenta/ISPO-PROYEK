"""Titik masuk: python -m sahabat (dipanggil oleh "Mulai SAHABAT.bat").

Tidak ada yang perlu diatur: memilih kamera, port kosong, dan resolusi sesuai kecepatan laptop.
"""
from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import numpy as np

from .config import SETTINGS_PATH, load, save, update_local

_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def find_free_port(preferred: int) -> int:
    """Port yang diinginkan bila kosong; kalau tidak, port kosong mana pun."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return s.getsockname()[1]
    raise OSError("tidak ada port kosong")


def claim_port(port: int) -> socket.socket | None:
    """Memegang port sejak awal, sebelum bagian yang lama (kamera, model, pengukuran).

    Klik dua kali kedua selama instans pertama masih menyala tidak akan mendapat port ini, jadi tidak
    ikut membuka kamera atau mengukur kecepatan. None bila port sudah dipegang.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        s.listen(128)
    except OSError:
        s.close()
        return None
    return s


def wait_until_running(port: int, timeout_s: float) -> bool:
    """Menunggu SAHABAT lain di port itu selesai menyala."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if already_running(port):
            return True
        time.sleep(0.5)
    return already_running(port)


def already_running(port: int, timeout: float = 0.5) -> bool:
    """Benar bila SAHABAT lain sudah menjawab di port itu."""
    try:
        with _NO_PROXY.open(f"http://127.0.0.1:{port}/api/ping", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")).get("app") == "sahabat"
    except Exception:
        return False


def find_browser() -> Path | None:
    """Edge lebih dulu (selalu ada di Windows 10/11), lalu Chrome."""
    roots = [os.environ.get(v) for v in ("ProgramFiles(x86)", "ProgramFiles", "LocalAppData")]
    for tail in (("Microsoft", "Edge", "Application", "msedge.exe"),
                 ("Google", "Chrome", "Application", "chrome.exe")):
        for root in roots:
            if root:
                exe = Path(root).joinpath(*tail)
                if exe.is_file():
                    return exe
    return None


def open_browser(url: str) -> None:
    if os.environ.get("SAHABAT_NO_BROWSER"):
        return
    exe = find_browser()
    if exe is not None:
        try:
            subprocess.Popen([str(exe), f"--app={url}", "--start-fullscreen"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    webbrowser.open(url)


def save_words_choice(on: bool, path: Path = SETTINGS_PATH) -> None:
    """Mengingat pilihan tombol "Kata kasar" di pengaturan laptop ini. Gagal menyimpan tidak boleh mengganggu."""
    try:
        update_local(path, words_enabled=bool(on))
    except OSError:
        pass


def _first_frame(source, wait_s: float = 6.0) -> np.ndarray:
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        got = source.read()
        if got is not None:
            return got[1]
        time.sleep(0.1)
    return np.zeros((480, 640, 3), dtype=np.uint8)      # belum ada kamera: ukur dengan frame hitam


def main() -> int:
    record = "--rekam" in sys.argv[1:]
    cfg = load(SETTINGS_PATH)
    sock = claim_port(cfg.port)
    if sock is None:                                   # port dipegang: SAHABAT lain, atau program lain
        if already_running(cfg.port):
            print("SAHABAT sudah berjalan. Membuka layarnya...")
            open_browser(f"http://127.0.0.1:{cfg.port}/")
            return 0
        print("SAHABAT sedang dinyalakan di jendela lain. Menunggu...")
        if wait_until_running(cfg.port, 120.0):
            open_browser(f"http://127.0.0.1:{cfg.port}/")
            return 0
        sock = claim_port(0)                           # port itu milik program lain: pakai port kosong mana pun
        if sock is None:
            print("Tidak ada port yang bisa dipakai.")
            return 1
    cfg = dataclasses.replace(cfg, port=sock.getsockname()[1])

    print("SAHABAT Sekolah sedang dinyalakan...")
    import uvicorn

    from .autotune import ensure_tuned
    from .server import build_engine, build_listener, create_app
    from .source import VideoSource

    source = VideoSource(cfg.camera, flip=cfg.camera_flip)
    source.start()
    frame = _first_frame(source)
    if cfg.imgsz is None or cfg.tuned_for is None:
        print("Mengukur kecepatan laptop ini (hanya sekali, sekitar 20 detik)...")
    cfg = ensure_tuned(cfg, frame)
    try:
        save(cfg, SETTINGS_PATH)
    except OSError:
        print("Peringatan: pengaturan tidak bisa disimpan (folder hanya-baca?). Lanjut tanpa menyimpan.")
    if cfg.slow:
        print("Peringatan: laptop ini terlalu lambat, deteksi bisa meleset.")

    engine = build_engine(cfg, source, record=record)
    if record and engine.recorder is not None and engine.recorder.active:
        print(f"MEREKAM: video uji disimpan di {engine.recorder.folder}")
        print("Rekaman berisi wajah orang. Hapus folder itu setelah penyetelan selesai.")
    engine.start()
    listener = None
    try:                                               # pendengar kata kasar boleh gagal; kamera tidak boleh ikut gagal
        recording = record and engine.recorder is not None and engine.recorder.active
        listener = build_listener(cfg, clip_dir=engine.recorder.folder / "suara" if recording else None,
                                  on_toggle=save_words_choice)
        listener.start()
        if recording:
            print("MEREKAM: potongan ucapan dari mikrofon juga disimpan di folder itu (subfolder suara).")
    except Exception as e:
        print(f"Peringatan: pendengar kata kasar tidak bisa dinyalakan ({e}); bagian kamera tetap jalan.")
        listener = None
    url = f"http://127.0.0.1:{cfg.port}/"
    server = uvicorn.Server(uvicorn.Config(create_app(engine, listener), host="127.0.0.1", port=cfg.port,
                                           log_level="warning"))

    def _open_when_ready() -> None:
        for _ in range(100):
            if already_running(cfg.port):
                break
            time.sleep(0.1)
        open_browser(url)
        if listener is not None:                       # tunggu model ucapan selesai dimuat, lalu laporkan sekali
            for _ in range(300):
                if listener.snapshot().status != "memuat":
                    break
                time.sleep(0.1)
            for _ in range(30):                        # mikrofon dibuka sesaat setelah model siap
                if listener.snapshot().status != "tanpa_mikrofon":
                    break
                time.sleep(0.1)
            print(listener.snapshot().label + (f"  (mikrofon: {listener.mic_name})" if listener.mic_name else ""), flush=True)

    threading.Thread(target=_open_when_ready, daemon=True).start()
    print(f"Siap di {url}  (resolusi {cfg.imgsz}). Tutup jendela ini untuk mematikan SAHABAT.")
    try:
        server.run(sockets=[sock])
    finally:
        if listener is not None:
            listener.stop()
        engine.stop()
        source.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
