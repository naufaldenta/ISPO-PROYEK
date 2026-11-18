"""Layar guru dari HP atau laptop lain di jaringan lokal (WiFi yang sama).

Mati secara bawaan. Bila dinyalakan, server kedua mendengar di semua antarmuka jaringan (port lan_port) dengan aplikasi
yang sama. Perangkat lain wajib memakai kode akses (dari kode QR atau diketik); laptop SAHABAT sendiri tidak perlu.
Perangkat lain hanya bisa melihat, mendengar alarm, menekan "Sudah diperiksa", dan membuka Riwayat; pengaturan
tetap hanya dari laptop SAHABAT.
"""
from __future__ import annotations

import ipaddress
import secrets
import socket
import threading

import cv2
import numpy as np

CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"      # tanpa 0/O dan 1/I supaya mudah diketik dari layar
CODE_LENGTH = 8
COOKIE = "sahabat_kode"


def new_code(length: int = CODE_LENGTH) -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(length))


def is_local(host: str | None) -> bool:
    """Permintaan dari laptop SAHABAT sendiri. "testclient" = klien uji bawaan Starlette, bukan alamat jaringan."""
    if not host:
        return False
    if host in ("localhost", "testclient"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def lan_addresses() -> list[str]:
    """Alamat IPv4 laptop ini di jaringan lokal, yang paling mungkin dipakai lebih dulu."""
    found: list[str] = []
    try:                                             # tidak mengirim apa pun; hanya menanyakan rute keluar
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        found += socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        pass
    out = []
    for ip in found:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if addr.version == 4 and not addr.is_loopback and not addr.is_link_local and ip not in out:
            out.append(ip)
    return out


def qr_png(text: str, module_px: int = 8, border: int = 4) -> bytes:
    """Kode QR hitam-putih sebagai PNG, tanpa pustaka tambahan (OpenCV)."""
    small = cv2.QRCodeEncoder.create().encode(text)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    small = np.where(small > 127, 255, 0).astype(np.uint8)
    small = cv2.copyMakeBorder(small, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    big = cv2.resize(small, None, fx=module_px, fy=module_px, interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", big)
    if not ok:
        raise ValueError("Kode QR tidak bisa dibuat.")
    return buf.tobytes()


class LanServer:
    """Server kedua untuk jaringan lokal. app diisi setelah create_app (aplikasinya butuh rujukan ke objek ini)."""

    def __init__(self, port: int, code: str | None = None, *, addresses=lan_addresses, server_factory=None,
                 host: str = "0.0.0.0"):
        self.port = port
        self.host = host                                # 0.0.0.0 = semua antarmuka; uji memakai 127.0.0.1
        self.code = (code or new_code()).upper()
        self.app = None
        self.enabled = False
        self.error = ""
        self.bound_port: int | None = None
        self.viewers = 0                                # layar jarak jauh yang sedang tersambung
        self._addresses = addresses
        self._server_factory = server_factory
        self._server = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ---- keadaan ----
    def urls(self) -> list[str]:
        if not self.enabled or self.bound_port is None:
            return []
        return [f"http://{ip}:{self.bound_port}/" for ip in self._addresses()]

    def status(self) -> dict:
        return {"hidup": self.enabled, "alamat": self.urls(), "kode": self.code, "port": self.bound_port,
                "galat": self.error, "penonton": self.viewers}

    def check(self, code: str | None) -> bool:
        return bool(code) and secrets.compare_digest(code.strip().upper(), self.code)

    def regenerate(self) -> str:
        """Kode baru; semua perangkat lain harus memindai ulang."""
        self.code = new_code()
        return self.code

    # ---- menyalakan / mematikan ----
    def set_enabled(self, on: bool) -> dict:
        with self._lock:
            if on and not self.enabled:
                self._start()
            elif not on and self.enabled:
                self._stop()
        return self.status()

    def _bind(self) -> socket.socket:
        last = None
        for port in (self.port, 0):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind((self.host, port))
                sock.listen(128)
                return sock
            except OSError as exc:
                last = exc
                sock.close()
        raise OSError(f"tidak ada port kosong ({last})")

    def _start(self) -> None:
        if self.app is None:
            raise RuntimeError("aplikasi belum dipasang")
        try:
            sock = self._bind()
        except OSError as exc:
            self.error = "Jaringan lokal tidak bisa dibuka: " + str(exc)
            return
        self.bound_port = sock.getsockname()[1]
        if self._server_factory is not None:
            self._server = self._server_factory(self.app)
        else:
            import uvicorn
            self._server = uvicorn.Server(uvicorn.Config(self.app, log_level="warning", lifespan="off"))
        self._thread = threading.Thread(target=self._server.run, kwargs={"sockets": [sock]}, name="jaringan-lokal",
                                        daemon=True)
        self._thread.start()
        self.enabled, self.error = True, ""

    def _stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server, self._thread = None, None
        self.enabled, self.bound_port, self.viewers = False, None, 0

    def stop(self) -> None:
        with self._lock:
            if self.enabled:
                self._stop()
