"""Menyimpan kata sandi kamera CCTV tanpa menuliskannya polos ke disk.

Di Windows dipakai DPAPI (CryptProtectData) dengan cakupan akun pengguna: berkas data/kamera.json yang disalin ke
laptop atau akun lain tidak bisa dibuka, dan kata sandinya cukup diisi ulang. Di sistem lain (hanya untuk tes dan
pengembangan) nilai disimpan sebagai base64 berawalan "b64:" — itu BUKAN enkripsi.
"""
from __future__ import annotations

import base64
import binascii
import sys

DPAPI = "dpapi:"
PLAIN = "b64:"


def _blob(data: bytes):
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    buf = ctypes.create_string_buffer(data, len(data))
    return DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf, DATA_BLOB


def _dpapi(data: bytes, protect: bool) -> bytes:
    import ctypes
    blob_in, _keep, DATA_BLOB = _blob(data)
    blob_out = DATA_BLOB()
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    if protect:
        ok = crypt32.CryptProtectData(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out))
    else:
        ok = crypt32.CryptUnprotectData(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out))
    if not ok:
        raise ValueError("Kata sandi tersimpan tidak bisa dibuka di akun Windows ini.")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def simpan(teks: str) -> str:
    data = teks.encode("utf-8")
    if sys.platform == "win32":
        return DPAPI + base64.b64encode(_dpapi(data, True)).decode("ascii")
    return PLAIN + base64.b64encode(data).decode("ascii")


def buka(nilai: str) -> str:
    """Kebalikan simpan(). ValueError bila nilai rusak atau berasal dari akun/laptop lain."""
    if not isinstance(nilai, str):
        raise ValueError("Kata sandi tersimpan rusak.")
    for prefix in (DPAPI, PLAIN):
        if nilai.startswith(prefix):
            try:
                raw = base64.b64decode(nilai[len(prefix):], validate=True)
            except (binascii.Error, ValueError):
                raise ValueError("Kata sandi tersimpan rusak.") from None
            if prefix == DPAPI:
                if sys.platform != "win32":
                    raise ValueError("Kata sandi tersimpan hanya bisa dibuka di Windows.")
                raw = _dpapi(raw, False)
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("Kata sandi tersimpan rusak.") from None
    raise ValueError("Kata sandi tersimpan rusak.")
