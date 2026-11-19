"""Server tiruan untuk menguji kamera jaringan tanpa kamera sungguhan."""
import http.server
import socket
import socketserver
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


class _Threading(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _serve(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


@pytest.fixture
def mjpeg_server():
    """Stream MJPEG (multipart) 640x360 bernomor sekitar 20 fps, seperti kamera murah atau aplikasi IP Webcam."""
    stopping = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            i = 0
            try:
                while not stopping.is_set():
                    img = np.full((360, 640, 3), 80, np.uint8)
                    cv2.putText(img, str(i), (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
                    jpg = cv2.imencode(".jpg", img)[1].tobytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                                     % len(jpg) + jpg + b"\r\n")
                    i += 1
                    time.sleep(0.05)
            except OSError:
                pass

    server = _Threading(("127.0.0.1", 0), Handler)
    _serve(server)
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_address[1]}/video")
    finally:
        stopping.set()
        server.shutdown()
        server.server_close()


@pytest.fixture
def fake_rtsp_server():
    """fake_rtsp_server(responder) -> url. responder(teks_permintaan) -> teks_jawaban, per permintaan."""
    servers = []

    def start(responder):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(8)
        sock.settimeout(0.2)
        stop = threading.Event()

        def loop():
            while not stop.is_set():
                try:
                    conn, _ = sock.accept()
                except OSError:
                    continue
                with conn:
                    conn.settimeout(2.0)
                    buf = b""
                    try:
                        while not stop.is_set():
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            buf += chunk
                            while b"\r\n\r\n" in buf:
                                req, buf = buf.split(b"\r\n\r\n", 1)
                                conn.sendall(responder(req.decode("utf-8", "replace")).encode())
                    except OSError:
                        pass

        threading.Thread(target=loop, daemon=True).start()
        servers.append((sock, stop))
        return f"rtsp://127.0.0.1:{sock.getsockname()[1]}/live"

    yield start
    for sock, stop in servers:
        stop.set()
        sock.close()


_PROFILES = """<?xml version="1.0"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
 xmlns:trt="http://www.onvif.org/ver10/media/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema"><s:Body>
<trt:GetProfilesResponse>
 <trt:Profiles token="main"><tt:Name>mainStream</tt:Name><tt:VideoEncoderConfiguration token="v1">
  <tt:Resolution><tt:Width>1920</tt:Width><tt:Height>1080</tt:Height></tt:Resolution></tt:VideoEncoderConfiguration>
 </trt:Profiles>
 <trt:Profiles token="sub"><tt:Name>subStream</tt:Name><tt:VideoEncoderConfiguration token="v2">
  <tt:Resolution><tt:Width>640</tt:Width><tt:Height>360</tt:Height></tt:Resolution></tt:VideoEncoderConfiguration>
 </trt:Profiles>
</trt:GetProfilesResponse></s:Body></s:Envelope>"""


@pytest.fixture
def fake_onvif_server():
    """Kamera ONVIF tiruan: GetCapabilities -> layanan Media, GetProfiles (1080p + 360p), GetStreamUri per token.
    Menolak dengan 401 bila nama pengguna di WS-Security bukan "admin"."""
    import re

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8", "replace")
            if "<Username>admin</Username>" not in body.replace("wsse:", "").replace("ns0:", ""):
                self.send_response(401)
                self.end_headers()
                return
            host = f"127.0.0.1:{self.server.server_address[1]}"
            if "GetCapabilities" in body:
                xml = (f'<?xml version="1.0"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
                       f'xmlns:tt="http://www.onvif.org/ver10/schema"><s:Body><GetCapabilitiesResponse>'
                       f'<Capabilities><tt:Media><tt:XAddr>http://{host}/onvif/Media</tt:XAddr></tt:Media>'
                       f'</Capabilities></GetCapabilitiesResponse></s:Body></s:Envelope>')
            elif "GetProfiles" in body:
                xml = _PROFILES
            elif "GetStreamUri" in body:
                token = re.search(r"ProfileToken>(\w+)<", body).group(1)
                xml = (f'<?xml version="1.0"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
                       f'xmlns:tt="http://www.onvif.org/ver10/schema"><s:Body><GetStreamUriResponse><MediaUri>'
                       f'<tt:Uri>rtsp://admin:bocor@127.0.0.1:554/{token}</tt:Uri></MediaUri>'
                       f'</GetStreamUriResponse></s:Body></s:Envelope>')
            else:
                self.send_response(500)
                self.end_headers()
                return
            data = xml.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/soap+xml")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = _Threading(("127.0.0.1", 0), Handler)
    _serve(server)
    try:
        yield SimpleNamespace(xaddr=f"http://127.0.0.1:{server.server_address[1]}/onvif/device_service")
    finally:
        server.shutdown()
        server.server_close()
