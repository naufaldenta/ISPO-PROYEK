import pytest

from sahabat import onvif
from sahabat.onvif import OnvifError, parse_probe_match, probe_message, stream_uri

PROBE_MATCH_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<SOAP-ENV:Envelope xmlns:SOAP-ENV="http://www.w3.org/2003/05/soap-envelope"
 xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
<SOAP-ENV:Body><d:ProbeMatches><d:ProbeMatch>
<wsa:EndpointReference><wsa:Address>urn:uuid:1234</wsa:Address></wsa:EndpointReference>
<d:Types>dn:NetworkVideoTransmitter</d:Types>
<d:Scopes>onvif://www.onvif.org/type/video_encoder onvif://www.onvif.org/name/Lorong%20A
 onvif://www.onvif.org/hardware/DS-2CD1023 onvif://www.onvif.org/location/city/hangzhou</d:Scopes>
<d:XAddrs>http://192.168.1.64/onvif/device_service http://[fe80::1]/onvif/device_service</d:XAddrs>
</d:ProbeMatch></d:ProbeMatches></SOAP-ENV:Body></SOAP-ENV:Envelope>"""


def test_probe_message_is_ws_discovery():
    msg = probe_message("uuid:1").decode()
    assert "Probe" in msg and "NetworkVideoTransmitter" in msg and "uuid:1" in msg


def test_parse_probe_match_reads_address_name_and_brand():
    assert parse_probe_match(PROBE_MATCH_SAMPLE) == {
        "host": "192.168.1.64", "xaddr": "http://192.168.1.64/onvif/device_service",
        "nama": "Lorong A", "merek": "DS-2CD1023"}
    assert parse_probe_match(b"<bukan xml") is None
    assert parse_probe_match(b"<a/>") is None


def test_stream_uri_picks_substream_and_strips_login(fake_onvif_server):
    sub = stream_uri(fake_onvif_server.xaddr, "admin", "x", substream=True)
    assert sub == "rtsp://127.0.0.1:554/sub"                     # login dari kamera dibuang
    assert stream_uri(fake_onvif_server.xaddr, "admin", "x", substream=False).endswith("/main")


def test_stream_uri_wrong_password(fake_onvif_server):
    with pytest.raises(OnvifError, match="Kata sandi salah"):
        stream_uri(fake_onvif_server.xaddr, "bukan", "x")


def test_stream_uri_camera_not_answering():
    with pytest.raises(OnvifError, match="tidak menjawab"):
        stream_uri("http://127.0.0.1:1/onvif/device_service", "admin", "x", timeout_s=1)


def test_discover_collects_unique_hosts(monkeypatch):
    monkeypatch.setattr(onvif, "_send_probes", lambda timeout_s: [PROBE_MATCH_SAMPLE, PROBE_MATCH_SAMPLE, b"x"])
    assert [c["host"] for c in onvif.discover(0.1)] == ["192.168.1.64"]


def test_onvif_ignores_system_proxy(fake_onvif_server, monkeypatch):   # Final #7
    import urllib.request
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"http": "http://127.0.0.1:1"})
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)
    monkeypatch.setattr(urllib.request, "_opener", None)          # opener global dibuat ulang dengan proxy tiruan
    assert stream_uri(fake_onvif_server.xaddr, "admin", "x").endswith("/sub")
