"""Layar guru dari HP atau laptop lain di jaringan lokal (WiFi yang sama).

Mati secara bawaan. Bila dinyalakan, server kedua mendengar di semua antarmuka jaringan (port lan_port) dengan aplikasi
yang sama. Perangkat lain wajib memakai kode akses (dari kode QR atau diketik); laptop SAHABAT sendiri tidak perlu.
Perangkat lain hanya bisa melihat, mendengar alarm, menekan "Sudah diperiksa", dan membuka Riwayat; pengaturan
tetap hanya dari laptop SAHABAT.
"""
from __future__ import annotations

import base64
import ipaddress
import re
import secrets
import socket
import subprocess
import sys
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


# Adaptor virtual (VMware, VirtualBox, Hyper-V/WSL, Docker, VPN) punya alamat 192.168.x.x juga, tetapi HP tidak bisa
# menjangkaunya. Alamat itu tidak ditampilkan supaya guru tidak memindai alamat yang salah.
VIRTUAL_ADAPTER = re.compile(r"vmware|virtualbox|vbox|vethernet|hyper-v|wsl|docker|loopback|bluetooth|npcap|"
                             r"tap-|tailscale|zerotier|wireguard|vpn", re.I)


def _route_address() -> str | None:
    """Alamat yang dipakai laptop untuk keluar ke jaringan. Tidak mengirim apa pun; hanya menanyakan rute."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return None


def _interfaces() -> dict[str, str] | None:
    """Alamat IPv4 -> nama adaptor yang sedang tersambung. None bila psutil tidak bisa dipakai."""
    try:
        import psutil
        stats = psutil.net_if_stats()
        out = {}
        for name, addrs in psutil.net_if_addrs().items():
            if name in stats and not stats[name].isup:
                continue
            for a in addrs:
                if a.family == socket.AF_INET:
                    out[a.address] = name
        return out
    except Exception:                                        # psutil tidak ada atau gagal membaca adaptor
        return None


def _usable(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.version == 4 and not addr.is_loopback and not addr.is_link_local and not addr.is_unspecified


def lan_addresses(route=_route_address, interfaces=_interfaces) -> list[str]:
    """Alamat IPv4 laptop ini di jaringan lokal, yang paling mungkin dipakai lebih dulu (WiFi/LAN, bukan VMware)."""
    first = route()
    names = interfaces()
    if names is None:                                        # tanpa nama adaptor: cara lama
        found = [first] if first else []
        try:
            found += socket.gethostbyname_ex(socket.gethostname())[2]
        except OSError:
            pass
        return list(dict.fromkeys(ip for ip in found if _usable(ip)))
    found = ([first] if first else []) + list(names)
    real = [ip for ip in found if _usable(ip) and ip in names and not VIRTUAL_ADAPTER.search(names[ip])]
    return list(dict.fromkeys(real))


# ---- Windows Firewall ----
# Tanpa aturan izin, Windows membuang koneksi dari HP tanpa pesan apa pun (terutama di WiFi berprofil "Publik"),
# sehingga HP hanya memuat terus. Aturan ini hanya membuka program Python SAHABAT, hanya TCP, dan hanya untuk
# perangkat di jaringan lokal yang sama (LocalSubnet). Pemasangannya selalu lewat persetujuan admin (UAC).
FIREWALL_RULE = "SAHABAT Sekolah (HP dan laptop lain)"


def _ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _encoded(script: str) -> str:
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def firewall_script_status(program: str) -> str:
    return "; ".join([
        f"$p = {_ps_quote(program)}",
        "if (-not @(Get-NetFirewallProfile | Where-Object { $_.Enabled }).Count) { 'tidak-perlu'; exit }",
        "$r = @(Get-NetFirewallApplicationFilter -Program $p -ErrorAction SilentlyContinue | "
        "Get-NetFirewallRule -ErrorAction SilentlyContinue | "
        "Where-Object { $_.Enabled -eq 'True' -and $_.Direction -eq 'Inbound' })",
        "if ($r | Where-Object { $_.Action -eq 'Block' }) { 'diblokir' } "
        "elseif ($r | Where-Object { $_.Action -eq 'Allow' }) { 'izin' } else { 'belum' }",
    ])


def firewall_script_allow(program: str) -> str:
    return "; ".join([
        f"$p = {_ps_quote(program)}",
        f"Remove-NetFirewallRule -DisplayName {_ps_quote(FIREWALL_RULE)} -ErrorAction SilentlyContinue",
        "Get-NetFirewallApplicationFilter -Program $p -ErrorAction SilentlyContinue | "
        "Get-NetFirewallRule -ErrorAction SilentlyContinue | "
        "Where-Object { $_.Direction -eq 'Inbound' -and $_.Action -eq 'Block' } | Disable-NetFirewallRule",
        f"New-NetFirewallRule -DisplayName {_ps_quote(FIREWALL_RULE)} -Direction Inbound -Action Allow "
        "-Program $p -Protocol TCP -RemoteAddress LocalSubnet -Profile Any | Out-Null",
    ])


def firewall_status(program: str | None = None) -> str:
    """izin / belum / diblokir / tidak-perlu (bukan Windows atau firewall mati) / tidak-diketahui."""
    if sys.platform != "win32":
        return "tidak-perlu"
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand",
                              _encoded(firewall_script_status(program or sys.executable))],
                             capture_output=True, text=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return "tidak-diketahui"
    word = out.stdout.strip().splitlines()[-1].strip() if out.stdout.strip() else ""
    return word if word in ("izin", "belum", "diblokir", "tidak-perlu") else "tidak-diketahui"


def allow_firewall(program: str | None = None) -> bool:
    """Meminta Windows memasang aturan izin. Windows menampilkan jendela persetujuan admin (UAC); True bila disetujui
    dan dijalankan. Hasilnya diperiksa ulang dengan firewall_status()."""
    if sys.platform != "win32":
        return False
    import ctypes
    args = "-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand " + _encoded(
        firewall_script_allow(program or sys.executable))
    return ctypes.windll.shell32.ShellExecuteW(None, "runas", "powershell.exe", args, None, 0) > 32


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
