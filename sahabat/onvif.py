"""Mencari IP camera di jaringan lokal (WS-Discovery) dan mengambil alamat stream-nya lewat ONVIF.

Hanya pustaka standar. Login ONVIF memakai WS-UsernameToken PasswordDigest. Jam kamera yang berbeda jauh dari jam
laptop bisa membuat login ditolak; pesan galatnya menyebut itu.
"""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import select
import socket
import time
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import unquote, urlsplit
from xml.sax.saxutils import escape

from .merek import pisah_login

MULTICAST = ("239.255.255.250", 3702)
# Kamera ada di jaringan lokal: proxy Windows sekolah tidak boleh dipakai (pengecualian <local> tidak mencakup
# alamat IP seperti 192.168.1.64, sehingga semua panggilan ONVIF akan gagal lewat proxy).
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))
WSSE = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
WSU = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
DIGEST = ("http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest")
B64 = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary"

MSG_PASSWORD = "Kata sandi salah (atau jam kamera berbeda jauh dari jam laptop)."
MSG_SILENT = "Kamera tidak menjawab. Periksa IP dan pastikan laptop di jaringan yang sama."
MSG_UNSUPPORTED = "Kamera tidak mendukung ONVIF; pakai Pilih merek."


class OnvifError(Exception):
    pass


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_all(root, name: str):
    return [el for el in root.iter() if _local(el.tag) == name]


# ---- WS-Discovery ----
def probe_message(message_id: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
        'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
        'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
        f'<e:Header><w:MessageID>{escape(message_id)}</w:MessageID>'
        '<w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
        '<w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>'
        '</e:Header><e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>'
        '</e:Envelope>').encode("utf-8")


def parse_probe_match(xml: bytes) -> dict | None:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    xaddrs = " ".join((el.text or "") for el in _find_all(root, "XAddrs")).split()
    xaddr = next((x for x in xaddrs if x.lower().startswith("http") and "[" not in x), None)
    if xaddr is None:
        return None
    scopes = " ".join((el.text or "") for el in _find_all(root, "Scopes")).split()

    def scope(kind: str) -> str:
        prefix = f"onvif://www.onvif.org/{kind}/"
        return next((unquote(s[len(prefix):]) for s in scopes if s.startswith(prefix)), "")

    return {"host": urlsplit(xaddr).hostname or "", "xaddr": xaddr, "nama": scope("name"),
            "merek": scope("hardware")}


def _send_probes(timeout_s: float) -> list[bytes]:
    """Satu-satunya bagian yang menyentuh jaringan multicast: probe dari tiap antarmuka LAN, kumpulkan jawaban."""
    from .jaringan import lan_addresses
    sockets = []
    for ip in lan_addresses() or ["0.0.0.0"]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.bind((ip, 0))
            if ip != "0.0.0.0":
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(ip))
            sock.sendto(probe_message(f"uuid:{uuid.uuid4()}"), MULTICAST)
            sock.setblocking(False)
            sockets.append(sock)
        except OSError:
            sock.close()
    replies: list[bytes] = []
    deadline = time.monotonic() + timeout_s
    try:
        while sockets and time.monotonic() < deadline:
            ready, _, _ = select.select(sockets, [], [], max(0.0, deadline - time.monotonic()))
            for sock in ready:
                try:
                    replies.append(sock.recv(65535))
                except OSError:
                    pass
    finally:
        for sock in sockets:
            sock.close()
    return replies


def discover(timeout_s: float = 3.0) -> list[dict]:
    found: dict[str, dict] = {}
    for reply in _send_probes(timeout_s):
        match = parse_probe_match(reply)
        if match and match["host"] and match["host"] not in found:
            found[match["host"]] = match

    def key(host: str):
        try:
            return (0, ipaddress.ip_address(host))
        except ValueError:
            return (1, host)
    return [found[h] for h in sorted(found, key=key)]


# ---- ONVIF SOAP ----
def _security(pengguna: str, sandi: str) -> str:
    nonce = uuid.uuid4().bytes
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + sandi.encode("utf-8")).digest()).decode()
    return (f'<Security s:mustUnderstand="1" xmlns="{WSSE}"><UsernameToken>'
            f'<Username>{escape(pengguna)}</Username><Password Type="{DIGEST}">{digest}</Password>'
            f'<Nonce EncodingType="{B64}">{base64.b64encode(nonce).decode()}</Nonce>'
            f'<Created xmlns="{WSU}">{created}</Created></UsernameToken></Security>')


def _call(url: str, body: str, pengguna: str, sandi: str, timeout_s: float) -> ET.Element:
    envelope = ('<?xml version="1.0" encoding="UTF-8"?>'
                '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
                f'<s:Header>{_security(pengguna, sandi)}</s:Header><s:Body>{body}</s:Body></s:Envelope>')
    request = urllib.request.Request(url, data=envelope.encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/soap+xml; charset=utf-8"})
    try:
        with _NO_PROXY.open(request, timeout=timeout_s) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace") if exc.fp else ""
        if exc.code == 401 or "NotAuthorized" in text:
            raise OnvifError(MSG_PASSWORD) from None
        raise OnvifError(MSG_UNSUPPORTED) from None
    except (urllib.error.URLError, OSError):
        raise OnvifError(MSG_SILENT) from None
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        raise OnvifError(MSG_UNSUPPORTED) from None
    if _find_all(root, "Fault"):
        raise OnvifError(MSG_PASSWORD if b"NotAuthorized" in data else MSG_UNSUPPORTED)
    return root


def _media_url(xaddr: str, pengguna: str, sandi: str, timeout_s: float) -> str:
    root = _call(xaddr, '<GetCapabilities xmlns="http://www.onvif.org/ver10/device/wsdl">'
                        '<Category>Media</Category></GetCapabilities>', pengguna, sandi, timeout_s)
    for media in _find_all(root, "Media"):
        for el in media.iter():
            if _local(el.tag) == "XAddr" and el.text:
                return el.text.strip()
    return xaddr


def stream_uri(xaddr: str, pengguna: str, sandi: str, substream: bool = True, timeout_s: float = 5.0) -> str:
    """Alamat RTSP kamera tanpa login (login disimpan terpisah di panel Kamera)."""
    media = _media_url(xaddr, pengguna, sandi, timeout_s)
    root = _call(media, '<GetProfiles xmlns="http://www.onvif.org/ver10/media/wsdl"/>', pengguna, sandi, timeout_s)
    profiles = []
    for prof in _find_all(root, "Profiles"):
        token = prof.get("token")
        if not token:
            continue
        width = next((int(el.text) for el in prof.iter() if _local(el.tag) == "Width" and (el.text or "").isdigit()), 0)
        height = next((int(el.text) for el in prof.iter() if _local(el.tag) == "Height" and (el.text or "").isdigit()), 0)
        profiles.append((width * height, token))
    if not profiles:
        raise OnvifError(MSG_UNSUPPORTED)
    profiles.sort()
    token = profiles[0][1] if substream else profiles[-1][1]
    root = _call(media, '<GetStreamUri xmlns="http://www.onvif.org/ver10/media/wsdl"><StreamSetup>'
                        '<Stream xmlns="http://www.onvif.org/ver10/schema">RTP-Unicast</Stream>'
                        '<Transport xmlns="http://www.onvif.org/ver10/schema"><Protocol>RTSP</Protocol></Transport>'
                        f'</StreamSetup><ProfileToken>{escape(token)}</ProfileToken></GetStreamUri>',
                 pengguna, sandi, timeout_s)
    uri = next((el.text.strip() for el in _find_all(root, "Uri") if el.text), "")
    if not uri.lower().startswith(("rtsp://", "rtsps://")):
        raise OnvifError(MSG_UNSUPPORTED)
    return pisah_login(uri)[0]
