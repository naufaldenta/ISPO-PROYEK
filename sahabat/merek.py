"""Alamat stream kamera CCTV per merek, dan penyamaran kata sandi di teks apa pun.

Pola alamat di bawah diambil dari dokumentasi pabrikan dan kebiasaan umum, dan di repo ini hanya diuji dengan server
tiruan. Kamera atau DVR tertentu bisa memakai jalur lain: bila gagal, pakai "Alamat lengkap" di panel Kamera.
"""
from __future__ import annotations

import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit

SKEMA_BOLEH = ("rtsp", "rtsps", "http", "https")

# channel False = merek itu tidak memakai nomor channel (satu kamera per alamat)
MEREK: dict[str, dict] = {
    "hikvision": {"nama": "Hikvision / HiLook", "port": 554, "channel": True},
    "dahua": {"nama": "Dahua", "port": 554, "channel": True},
    "uniview": {"nama": "Uniview", "port": 554, "channel": True},
    "tapo": {"nama": "TP-Link Tapo / VIGI", "port": 554, "channel": False},
    "ezviz": {"nama": "EZVIZ", "port": 554, "channel": True},
    "imou": {"nama": "Imou", "port": 554, "channel": True},
    "xmeye": {"nama": "DVR/NVR murah (XMeye, iCSee)", "port": 554, "channel": True},
    "ipwebcam": {"nama": "HP Android (aplikasi IP Webcam)", "port": 8080, "channel": False},
}


def _q(text: str) -> str:
    return quote(text or "", safe="")


def _auth(pengguna: str, sandi: str) -> str:
    if not pengguna and not sandi:
        return ""
    return _q(pengguna) + (":" + _q(sandi) if sandi else "") + "@"


def alamat_merek(merek: str, host: str, port: int, channel: int, substream: bool, pengguna: str, sandi: str) -> str:
    """Alamat lengkap (dengan login) untuk satu channel kamera merek tertentu."""
    if merek not in MEREK:
        raise ValueError("Merek kamera tidak dikenal.")
    base = f"{host}:{int(port)}"
    auth = _auth(pengguna, sandi)
    ch = int(channel)
    if merek == "hikvision":
        return f"rtsp://{auth}{base}/Streaming/Channels/{ch}0{2 if substream else 1}"
    if merek in ("dahua", "imou"):
        return f"rtsp://{auth}{base}/cam/realmonitor?channel={ch}&subtype={1 if substream else 0}"
    if merek == "uniview":
        return f"rtsp://{auth}{base}/unicast/c{ch}/s{1 if substream else 0}/live"
    if merek == "tapo":
        return f"rtsp://{auth}{base}/stream{2 if substream else 1}"
    if merek == "ezviz":
        return f"rtsp://{auth}{base}/h264/ch{ch}/{'sub' if substream else 'main'}/av_stream"
    if merek == "xmeye":        # login ada di jalur, bukan sebelum "@"
        return (f"rtsp://{base}/user={_q(pengguna)}&password={_q(sandi)}&channel={ch}"
                f"&stream={1 if substream else 0}.sdp?real_stream")
    return f"http://{auth}{base}/video"                      # ipwebcam: MJPEG


def _host_part(netloc: str) -> str:
    return netloc.rsplit("@", 1)[-1]


def pasang_login(url: str, pengguna: str, sandi: str) -> str:
    """Menyisipkan login ke alamat lengkap yang belum memuat login."""
    if not pengguna and not sandi:
        return url
    parts = urlsplit(url)
    return urlunsplit(parts._replace(netloc=_auth(pengguna, sandi) + _host_part(parts.netloc)))


def pisah_login(url: str) -> tuple[str, str, str]:
    """(alamat tanpa login, pengguna, sandi) — login di URL yang ditempel guru dipindah ke isian terpisah.

    Tidak memakai urlsplit untuk login: kata sandi yang ditempel mentah bisa memuat "/", "#", atau "?", sehingga
    batas host diambil dari "@" terakhir, bukan dari "/" pertama.
    """
    scheme, sep, rest = url.partition("://")
    at = rest.rfind("@")
    if not sep or at < 0:
        return url, "", ""
    userinfo, hostpath = rest[:at], rest[at + 1:]
    if not hostpath or hostpath[0] in "/?#":
        return url, "", ""
    pengguna, _, sandi = userinfo.partition(":")
    return f"{scheme}://{hostpath}", unquote(pengguna), unquote(sandi)


SANDI_MARK = "{SANDI}"                 # tempat kata sandi di query alamat (?pwd=...), diisi saat stream dibuka
_QUERY_SECRET = re.compile(r"([?&](?:password|passwd|pwd|pass)=)([^&]*)", re.I)   # "#" mentah ikut sandi


def pisah_sandi_query(url: str) -> tuple[str, str | None]:
    """Kata sandi di query (misalnya ?user=admin&pwd=rahasia) diganti {SANDI}; (alamat, sandi | None)."""
    match = _QUERY_SECRET.search(url)
    if not match or match.group(2) == SANDI_MARK:
        return url, None
    return url[:match.start(2)] + SANDI_MARK + url[match.end(2):], unquote(match.group(2))


def isi_sandi_query(url: str, sandi: str) -> str:
    return url.replace(SANDI_MARK, _q(sandi))


_USERINFO = re.compile(r"([a-z][a-z0-9+.\-]*://[^\s:/@]*):[^\s@/]*@", re.I)
_QUERY_PASS = re.compile(r"\b(password|passwd|pwd|pass)=[^&\s]*", re.I)


def samarkan(teks: str) -> str:
    """Kata sandi di URL mana pun di teks diganti •••• (pesan galat, log, tampilan alamat)."""
    teks = _USERINFO.sub(r"\1:••••@", str(teks))
    return _QUERY_PASS.sub(r"\1=••••", teks)
