"""Perhitungan pemutar inspeksi (sahabat/web/inspeksi-hitung.js) diuji lewat Node bila tersedia."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

HITUNG = Path(__file__).resolve().parent.parent / "sahabat" / "web" / "inspeksi-hitung.js"
NODE = shutil.which("node")


def run(expr: str):
    code = f"const h = require({json.dumps(str(HITUNG))}); process.stdout.write(JSON.stringify({expr}));"
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="butuh Node.js")
def test_event_marker_follows_frames_not_time():                                    # Final #2
    """Frame tidak merata (siaga 2 fps lalu penuh 10 fps): penanda merah harus di posisi frame kejadian pada garis
    waktu yang bergerak per frame, bukan di posisi detiknya."""
    waktu = [i * 0.5 for i in range(12)] + [6.0 + i * 0.1 for i in range(1, 51)]     # 12 frame lambat, 50 cepat
    marks = run(f"h.markerPositions({json.dumps(waktu)}, 6.5, [], 0.8)")
    event_index = next(i for i, t in enumerate(waktu) if t >= 6.5 - 1e-6)
    assert marks["kejadian"] == pytest.approx(event_index / (len(waktu) - 1))
    assert abs(marks["kejadian"] - 6.5 / waktu[-1]) > 0.1                          # beda jauh dari posisi detik


@pytest.mark.skipif(NODE is None, reason="butuh Node.js")
def test_score_markers_and_event_index():
    overlay = [None, {"skor": {"jatuh": True, "kekerasan": 0}}, {"skor": {"jatuh": False, "kekerasan": 0.85}},
               {"skor": {"jatuh": False, "kekerasan": 0.5}}]
    marks = run(f"h.markerPositions([0, 0.1, 0.2, 0.3], 0.2, {json.dumps(overlay)}, 0.8)")
    assert marks["skor"] == pytest.approx([1 / 3, 2 / 3]) and marks["kejadian"] == pytest.approx(2 / 3)
    assert run("h.frameAt([0, 0.1, 0.2, 0.3], 0.25)") == 2 and run("h.frameAt([0, 0.1], -1)") == 0
    assert run("h.markerPositions([0], null, null, 0.8)") == {"kejadian": None, "skor": []}
