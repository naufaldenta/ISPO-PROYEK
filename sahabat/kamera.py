"""Daftar kamera CCTV milik laptop ini (data/kamera.json, tidak ikut di-commit).

Setiap kamera dipilih sebagai sumber dengan spesifikasi "ipcam:<id>". Alamat lengkap (dengan login) hanya dibangun
di sini lewat resolve(), saat sumber video membuka stream; pengaturan, status layar, dan log hanya memuat
"ipcam:<id>" atau alamat yang disamarkan. Kata sandi disimpan terenkripsi (sahabat.rahasia).

Jenis kamera:
- "merek": IP camera / channel DVR-NVR dari pola merek (sahabat.merek);
- "alamat": alamat lengkap rtsp/http dari teknisi atau dari pencarian ONVIF;
- "perangkat": capture card USB (keluaran HDMI DVR) yang terbaca seperti webcam ("camera:<id>").
Ketiganya boleh memotong satu ubin dari tampilan mosaik DVR (potong = [grid, ubin]).
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .profil_kamera import posisi_bersih
from .merek import (MEREK, SANDI_MARK, SKEMA_BOLEH, alamat_merek, isi_sandi_query, pasang_login, pisah_login,
                    pisah_sandi_query, samarkan)
from .rahasia import buka, simpan

IPCAM_PREFIX = "ipcam:"
JENIS = ("merek", "alamat", "perangkat", "video")
MAX_NAMA = 40
_ID = re.compile(r"^[a-z0-9]{1,12}$")
_HOST = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.\-]{0,251}[A-Za-z0-9])?$")


PASSWORD_UNREADABLE = "Kata sandi kamera perlu diisi ulang di panel Kamera (tekan Ubah)."


@dataclass(frozen=True)
class Stream:
    url: str                                # alamat lengkap dengan login, atau "camera:<id>"; "" bila tidak dibuka
    potong: tuple[int, int] | None          # (grid, ubin) dari tampilan mosaik DVR; ubin mulai 0
    galat: str = ""                         # alasan stream tidak boleh dibuka (misalnya sandi perlu diisi ulang)


def _int(value, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} tidak valid.")
    try:
        number = int(value)
    except ValueError:
        raise ValueError(f"{name} harus angka.") from None
    if not lo <= number <= hi or (isinstance(value, float) and value != number):
        raise ValueError(f"{name} harus antara {lo} dan {hi}.")
    return number


def _text(value, name: str, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{name} tidak valid.")
    return value


def _host(value) -> str:
    host = _text(value, "Alamat IP", 253).strip()
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
        return f"[{ip}]" if ip.version == 6 else str(ip)
    except ValueError:
        pass
    if not _HOST.match(host):
        raise ValueError("Alamat IP kamera tidak valid (contoh: 192.168.1.64).")
    return host


def _potong(value) -> tuple[int, int] | None:
    if value in (None, [], ""):
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Pilihan ubin DVR tidak valid.")
    grid = _int(value[0], "Ukuran grid DVR", 2, 4)
    ubin = _int(value[1], "Nomor ubin", 0, grid * grid - 1)
    return grid, ubin


# Nilai isian form yang tidak ada di berkas lama (sama dengan bawaan _clean).
FORM_DEFAULTS = {"merek": "", "host": "", "port": 0, "channel": 1, "substream": True, "alamat": "", "perangkat": "",
                 "video": "", "pengguna": "", "potong": None}


class KameraStore:
    def __init__(self, path: Path, video_dir: Path | None = None) -> None:
        from .config import VIDEO_DIR
        self.path = Path(path)
        self.video_dir = Path(video_dir) if video_dir is not None else VIDEO_DIR   # jenis "video" (simulasi)
        self._lock = threading.RLock()
        self._items: list[dict] = self._load()

    # ---- berkas ----
    def _load(self) -> list[dict]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, ValueError):
            self._backup_broken()
            return []
        items = raw.get("kamera") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            self._backup_broken()
            return []
        return [it for it in items if isinstance(it, dict) and isinstance(it.get("id"), str)
                and _ID.match(it["id"]) and it.get("jenis") in JENIS]

    def _backup_broken(self) -> None:
        """Berkas rusak disalin dulu supaya daftar lama tidak hilang begitu saja saat disimpan ulang."""
        try:
            shutil.copyfile(self.path, self.path.with_suffix(".rusak.json"))
        except OSError:
            pass

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as f:
                tmp = f.name
                json.dump({"kamera": self._items}, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)
        finally:
            if tmp is not None and os.path.exists(tmp):
                os.unlink(tmp)

    def _find(self, cam_id) -> dict | None:
        return next((it for it in self._items if it["id"] == cam_id), None)

    # ---- kata sandi ----
    @staticmethod
    def _password(item: dict) -> tuple[str, bool]:
        """(sandi, rusak). Sandi yang tidak bisa dibuka (akun Windows lain) dianggap kosong."""
        stored = item.get("sandi") or ""
        if not stored:
            return "", False
        try:
            return buka(stored), False
        except ValueError:
            return "", True

    # ---- validasi ----
    def _clean(self, data: dict) -> tuple[dict, str | None]:
        """(isian bersih tanpa sandi, sandi baru | None = pertahankan sandi lama)."""
        if not isinstance(data, dict):
            raise ValueError("Isian kamera tidak valid.")
        nama = _text(data.get("nama"), "Nama kamera", 200).strip()
        if not nama or len(nama) > MAX_NAMA:
            raise ValueError(f"Nama kamera wajib diisi, paling panjang {MAX_NAMA} huruf.")
        jenis = data.get("jenis")
        if jenis not in JENIS:
            raise ValueError("Jenis kamera tidak valid.")
        pengguna = _text(data.get("pengguna"), "Nama pengguna", 64).strip()
        sandi = data.get("sandi")
        if sandi is not None:
            sandi = _text(sandi, "Kata sandi", 128)
        out = {"nama": nama, "jenis": jenis, "merek": "", "host": "", "port": 0, "channel": 1, "substream": True,
               "alamat": "", "perangkat": "", "video": "", "pengguna": pengguna, "potong": _potong(data.get("potong")),
               "posisi": posisi_bersih(data.get("posisi"))}
        if jenis == "merek":
            merek = data.get("merek")
            if merek not in MEREK:
                raise ValueError("Pilih merek kamera dari daftar.")
            out.update(merek=merek, host=_host(data.get("host")),
                       port=_int(data.get("port") or MEREK[merek]["port"], "Port", 1, 65535),
                       channel=_int(data.get("channel") or 1, "Channel", 1, 64) if MEREK[merek]["channel"] else 1,
                       substream=bool(data.get("substream", True)))
        elif jenis == "alamat":
            alamat = _text(data.get("alamat"), "Alamat", 1000).strip()
            clean, url_user, url_pass = pisah_login(alamat)
            clean, query_pass = pisah_sandi_query(clean)
            parts = urlsplit(clean)
            try:
                parts.port
            except ValueError:
                raise ValueError("Port di alamat tidak valid.") from None
            if parts.scheme.lower() not in SKEMA_BOLEH or not parts.hostname or "@" in parts.netloc:
                raise ValueError("Alamat harus diawali rtsp:// atau http:// lalu alamat kamera.")
            out["alamat"] = clean
            if url_user or url_pass:
                out["pengguna"] = url_user
                sandi = url_pass
            if query_pass is not None:
                sandi = query_pass
        elif jenis == "video":
            from .devices import safe_video_name
            name = safe_video_name(data.get("video"))
            if name is None:
                raise ValueError("Pilih video dari folder video.")
            out.update(video=name, pengguna="")
            sandi = ""
        else:
            perangkat = _text(data.get("perangkat"), "Perangkat", 1000)
            if not perangkat.startswith("camera:"):
                raise ValueError("Pilih capture card dari daftar perangkat.")
            out["perangkat"] = perangkat
        return out, sandi

    # ---- alamat ----
    def _url(self, item: dict, sandi: str) -> str:
        if item["jenis"] == "merek":
            return alamat_merek(item["merek"], item["host"], item["port"], item["channel"], item["substream"],
                                item.get("pengguna", ""), sandi)
        if item["jenis"] == "alamat":
            if SANDI_MARK in item["alamat"]:          # kata sandi di query, bukan sebelum "@"
                return pasang_login(isi_sandi_query(item["alamat"], sandi), item.get("pengguna", ""), "")
            return pasang_login(item["alamat"], item.get("pengguna", ""), sandi)
        if item["jenis"] == "video":
            return str(self.video_dir / item["video"])
        return item["perangkat"]

    def _public(self, item: dict) -> dict:
        sandi, rusak = self._password(item)
        shown = ("Video: " + item.get("video", "")) if item["jenis"] == "video" else \
            samarkan(self._url(item, "x" if (sandi or rusak) else ""))
        potong = item.get("potong")
        return {"id": item["id"], "nama": item["nama"], "jenis": item["jenis"], "merek": item.get("merek", ""),
                "host": item.get("host", ""), "port": item.get("port", 0), "channel": item.get("channel", 1),
                "substream": item.get("substream", True), "alamat_tampil": shown,
                "alamat_edit": item.get("alamat", "") if item["jenis"] == "alamat" else "",
                "pengguna": item.get("pengguna", ""), "ada_sandi": bool(item.get("sandi")), "sandi_rusak": rusak,
                "perangkat": item.get("perangkat", ""), "potong": list(potong) if potong else None,
                "video": item.get("video", ""), "aktif": bool(item.get("aktif", True)),
                "prioritas": bool(item.get("prioritas", False)), "zona": item.get("zona"),
                "posisi": posisi_bersih(item.get("posisi")), "rev": int(item.get("rev", 0))}

    # ---- API ----
    def list_public(self) -> list[dict]:
        with self._lock:
            return [self._public(it) for it in self._items]

    def get_public(self, cam_id: str) -> dict | None:
        with self._lock:
            item = self._find(cam_id)
            return self._public(item) if item else None

    def save(self, data: dict) -> dict:
        with self._lock:
            clean, sandi = self._clean(data)
            cam_id = data.get("id")
            if cam_id:
                item = self._find(cam_id)
                if item is None:
                    raise ValueError("Kamera tidak ditemukan. Muat ulang daftar kamera.")
                old_sandi = item.get("sandi", "")
                kept = {k: item[k] for k in ("aktif", "prioritas", "zona") if k in item}   # hanya lewat set_*
                same_stream = sandi is None and all(
                    (list(v) if isinstance(v, tuple) else v) == item.get(k, FORM_DEFAULTS.get(k))
                    for k, v in clean.items() if k != "posisi")
                # naik tiap simpan (Mode CCTV menyambung ulang), kecuali yang berubah hanya posisi kamera
                rev = int(item.get("rev", 0)) + (0 if same_stream else 1)
                item.clear()
                item.update(id=cam_id, **clean, **kept, rev=rev)
                item["sandi"] = old_sandi if sandi is None else (simpan(sandi) if sandi else "")
            else:
                cam_id = "k" + secrets.token_hex(3)
                while self._find(cam_id):
                    cam_id = "k" + secrets.token_hex(3)
                item = {"id": cam_id, **clean, "sandi": simpan(sandi) if sandi else "", "aktif": True,
                        "prioritas": False, "zona": None}
                self._items.append(item)
            if item.get("potong"):
                item["potong"] = list(item["potong"])
            self._write()
            return self._public(item)

    def delete(self, cam_id: str) -> bool:
        with self._lock:
            item = self._find(cam_id)
            if item is None:
                return False
            self._items.remove(item)
            self._write()
            return True

    def resolve(self, spec: str) -> Stream | None:
        if not isinstance(spec, str) or not spec.startswith(IPCAM_PREFIX):
            return None
        with self._lock:
            item = self._find(spec[len(IPCAM_PREFIX):])
            if item is None:
                return None
            sandi, rusak = self._password(item)
            potong = tuple(item["potong"]) if item.get("potong") else None
            if item["jenis"] == "video" and not (self.video_dir / item.get("video", "")).is_file():
                return Stream("", potong, f"Video {item.get('video', '')} tidak ada di folder video.")
            if rusak:                       # mencoba login tanpa sandi berulang kali bisa mengunci akun kamera
                return Stream("", potong, PASSWORD_UNREADABLE)
            return Stream(self._url(item, sandi), potong)

    def set_flags(self, cam_id: str, aktif: bool | None = None, prioritas: bool | None = None) -> dict:
        """Tanda Mode CCTV: ikut dinilai (aktif) dan didahulukan penjadwal (prioritas)."""
        with self._lock:
            item = self._find(cam_id)
            if item is None:
                raise ValueError("Kamera tidak ditemukan.")
            if aktif is not None:
                item["aktif"] = bool(aktif)
            if prioritas is not None:
                item["prioritas"] = bool(prioritas)
            self._write()
            return self._public(item)

    def set_zone(self, cam_id: str, zona) -> dict:
        """Zona pantau kamera ini (titik ternormalisasi dari sahabat.zone.normalize) atau None."""
        with self._lock:
            item = self._find(cam_id)
            if item is None:
                raise ValueError("Kamera tidak ditemukan.")
            item["zona"] = [list(map(float, p)) for p in zona] if zona else None
            self._write()
            return self._public(item)

    def active(self) -> list[dict]:
        with self._lock:
            return [self._public(it) for it in self._items if it.get("aktif", True)]

    def catalog(self) -> list[dict]:
        with self._lock:
            return [{"id": IPCAM_PREFIX + it["id"], "name": it["nama"]} for it in self._items]

    def form_stream(self, data: dict) -> Stream:
        """Stream dari isi formulir tanpa menyimpan (Tes sambungan). Sandi kosong + id = pakai sandi tersimpan."""
        with self._lock:
            clean, sandi = self._clean(data)
            if not sandi and isinstance(data, dict) and data.get("id"):
                item = self._find(data["id"])
                # sandi tersimpan hanya dipakai untuk kamera yang sama, supaya tidak terkirim ke alamat lain
                if item is not None and self._url(clean, "") == self._url(item, ""):
                    sandi, _ = self._password(item)
            return Stream(self._url(clean, sandi or ""), clean["potong"])


_DEFAULT: KameraStore | None = None
_DEFAULT_LOCK = threading.Lock()


def default_store() -> KameraStore:
    """Satu daftar kamera bersama untuk aplikasi (data/kamera.json), supaya sumber video dan panel Kamera tidak
    memegang salinan yang berbeda."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            from .config import KAMERA_PATH
            _DEFAULT = KameraStore(KAMERA_PATH)
        return _DEFAULT
