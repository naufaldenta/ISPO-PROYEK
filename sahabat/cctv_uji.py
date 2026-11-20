"""Tombol "Tes sambungan" di panel Kamera: membuka stream sebentar, lalu melaporkan ukuran, laju, dan cuplikan, atau
alasan gagal dalam bahasa yang dipahami guru.

Untuk RTSP, permintaan DESCRIBE dikirim lebih dulu (dengan login Basic/Digest bila diminta kamera). OpenCV tidak
membedakan kata sandi salah dari jalur salah, padahal jawaban RTSP 401/404 membedakannya.
Hasil tidak pernah memuat alamat stream, jadi kata sandi tidak bisa ikut terkirim ke browser.
"""
from __future__ import annotations

import base64
import hashlib
import re
import secrets
import socket
import time
from urllib.parse import unquote, urlsplit, urlunsplit

import cv2

from .kamera import Stream
from .source import crop_tile, open_stream, source_kind

PESAN = {
    "sandi": "Kata sandi atau nama pengguna salah.",
    "jalur": "Alamat stream tidak ditemukan di kamera. Coba merek lain, channel lain, atau substream.",
    "tak_terjangkau": "Kamera tidak menjawab. Periksa IP, kabel/WiFi, dan pastikan laptop di jaringan yang sama.",
    "lain": "Kamera menolak sambungan. Coba substream atau Alamat lengkap.",
    "dipakai": "Webcam / capture card tidak bisa dibuka. Mungkin sedang dipakai Mode Pameran atau aplikasi lain: "
               "tutup pemakainya, lalu tekan Tes sambungan lagi.",
}
UNREACHABLE_CODES = {10051, 10060, 10065, 51, 65, 101, 113}   # jaringan/host tidak terjangkau (Windows dan POSIX)
READ_S = 3.0                       # lama membaca untuk mengukur laju
PREVIEW_WIDTH = 480


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _auth_header(challenges: list[str], user: str, password: str, uri: str) -> str | None:
    def answerable(challenge: str) -> bool:
        algo = re.search(r'algorithm\s*=\s*"?([\w-]+)', challenge, re.IGNORECASE)
        return algo is None or algo.group(1).lower() in ("md5", "md5-sess")
    # Kamera bisa menawarkan beberapa tantangan (misalnya SHA-256 lalu MD5): pilih yang bisa dijawab.
    digest = next((c for c in challenges if c.lower().startswith("digest") and answerable(c)), None)
    if digest is not None:
        # Nilai berkutip boleh berisi koma (misalnya realm="IP Camera, Lobi").
        fields = {k.lower(): (a if a else b)
                  for k, a, b in re.findall(r'(\w+)\s*=\s*(?:"([^"]*)"|([^,\s]*))', digest[len("digest"):])}
        realm, nonce = fields.get("realm", ""), fields.get("nonce", "")
        algorithm = fields.get("algorithm", "")
        cnonce, nc = secrets.token_hex(8), "00000001"
        ha1 = _md5(f"{user}:{realm}:{password}")
        if algorithm.lower() == "md5-sess":
            ha1 = _md5(f"{ha1}:{nonce}:{cnonce}")
        ha2 = _md5(f"DESCRIBE:{uri}")
        parts = [f'username="{user}"', f'realm="{realm}"', f'nonce="{nonce}"', f'uri="{uri}"']
        if "auth" in [q.strip() for q in fields.get("qop", "").split(",")]:
            parts += [f'response="{_md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}")}"', "qop=auth", f"nc={nc}",
                      f'cnonce="{cnonce}"']
        else:
            parts.append(f'response="{_md5(f"{ha1}:{nonce}:{ha2}")}"')
            if algorithm.lower() == "md5-sess":
                parts.append(f'cnonce="{cnonce}"')
        if algorithm:
            parts.append(f"algorithm={algorithm}")
        if "opaque" in fields:
            parts.append(f'opaque="{fields["opaque"]}"')        # wajib dikembalikan apa adanya bila diberi
        return "Digest " + ", ".join(parts)
    if any(c.lower().startswith("basic") for c in challenges):
        return "Basic " + base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return None


def _describe(sock: socket.socket, uri: str, cseq: int, auth: str | None) -> tuple[int, list[str]]:
    lines = [f"DESCRIBE {uri} RTSP/1.0", f"CSeq: {cseq}", "Accept: application/sdp", "User-Agent: SAHABAT"]
    if auth:
        lines.append("Authorization: " + auth)
    sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("utf-8"))
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    head = buf.split(b"\r\n\r\n", 1)[0].decode("utf-8", "replace").split("\r\n")
    match = re.match(r"RTSP/1\.\d\s+(\d{3})", head[0] if head else "")
    if not match:
        raise ValueError("bukan jawaban RTSP")
    challenges = [h.split(":", 1)[1].strip() for h in head[1:] if h.lower().startswith("www-authenticate:")]
    return int(match.group(1)), challenges


def rtsp_probe(url: str, timeout_s: float = 3.0) -> str:
    """ok / sandi / jalur / tak_terjangkau / lain, dari jawaban kamera atas DESCRIBE."""
    parts = urlsplit(url)
    if parts.scheme.lower() != "rtsp":
        return "ok"                                   # rtsps butuh TLS: biarkan OpenCV yang mencoba
    try:
        user, password = unquote(parts.username or ""), unquote(parts.password or "")
        host, port = parts.hostname or "", parts.port or 554
    except ValueError:                                # port bukan angka (alamat atau sandi mentah rusak)
        return "lain"
    uri = urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[-1]))
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            code, challenges = _describe(sock, uri, 1, None)
            if code == 401 and user:
                auth = _auth_header(challenges, user, password, uri)
                if auth is not None:
                    try:
                        code, _ = _describe(sock, uri, 2, auth)
                    except (OSError, ValueError):     # sebagian kamera menutup sambungan sesudah 401
                        with socket.create_connection((host, port), timeout=timeout_s) as again:
                            again.settimeout(timeout_s)
                            code, _ = _describe(again, uri, 2, auth)
    except (socket.timeout, ConnectionError, socket.gaierror):
        return "tak_terjangkau"
    except OSError as exc:
        if exc.errno in UNREACHABLE_CODES or getattr(exc, "winerror", None) in UNREACHABLE_CODES:
            return "tak_terjangkau"
        return "lain"
    except ValueError:
        return "lain"
    if 200 <= code < 300:
        return "ok"
    return {401: "sandi", 403: "sandi", 404: "jalur"}.get(code, "lain")


def _reachable(url: str, timeout_s: float) -> bool:
    parts = urlsplit(url)
    try:
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
        socket.create_connection((parts.hostname or "", port), timeout=timeout_s).close()
        return True
    except (OSError, ValueError):
        return False


def _fail(key: str) -> dict:
    return {"ok": False, "pesan": PESAN[key], "lebar": 0, "tinggi": 0, "fps": 0.0, "cuplikan": ""}


def uji_sambungan(stream: Stream, timeout_s: float = 10.0, opener=open_stream) -> dict:
    url = stream.url
    kind = source_kind(url)
    started = time.monotonic()
    if kind == "jaringan":
        scheme = urlsplit(url).scheme.lower()
        if scheme == "rtsp":
            result = rtsp_probe(url, timeout_s=min(3.0, timeout_s))
            if result != "ok":
                return _fail(result)
        elif scheme in ("http", "https") and not _reachable(url, min(3.0, timeout_s)):
            return _fail("tak_terjangkau")
        try:
            cap = opener(url)
        except cv2.error:                       # backend OpenCV menolak alamat ini: jawab ramah, bukan galat 500
            return _fail("lain")
    elif kind == "webcam":
        from .source import VideoSource
        try:
            cap = VideoSource(url)._open()
        except cv2.error:
            return _fail("dipakai")
    else:
        return _fail("lain")
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        return _fail("dipakai" if kind == "webcam" else "lain")
    try:
        times, first = [], None
        deadline = min(started + timeout_s, time.monotonic() + READ_S)
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            times.append(time.monotonic())
            if first is None:
                first = frame
    except cv2.error:
        times, first = [], None
    finally:
        cap.release()
    if first is None:
        return _fail("lain")
    if stream.potong is not None:
        first = crop_tile(first, *stream.potong)
    h, w = first.shape[:2]
    fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
    small = first if w <= PREVIEW_WIDTH else cv2.resize(first, (PREVIEW_WIDTH, round(h * PREVIEW_WIDTH / w)))
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    preview = base64.b64encode(jpg.tobytes()).decode("ascii") if ok else ""
    return {"ok": True, "pesan": f"Tersambung: {w}×{h}, {fps:.0f} fps", "lebar": w, "tinggi": h,
            "fps": round(fps, 1), "cuplikan": preview}
