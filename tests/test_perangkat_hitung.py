"""Memilih perangkat hitung Mode CCTV (GPU tercepat atau CPU) dan mengukurnya."""
import numpy as np
import pytest

from sahabat import perangkat_hitung as ph
from sahabat.config import LOCAL_FIELDS, Config
from sahabat.perangkat_hitung import HITUNG, kapasitas, pilih, ukur

F = np.zeros((48, 64, 3), np.uint8)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture(autouse=True)
def fake_clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(ph, "_clock", clock)
    return clock


class FakePose:
    """detect() memajukan jam tiruan sebanyak ms."""

    def __init__(self, ms, nama="CPU"):
        self.ms, self.nama, self.calls = ms, nama, 0

    def detect(self, cam_id, frame):
        self.calls += 1
        ph._clock.t += self.ms / 1000.0
        return []


def fake_factory(ms_by_id):
    def make(device_id):
        if device_id not in ms_by_id:
            raise RuntimeError("tidak ada adaptor")
        return FakePose(ms_by_id[device_id], nama=f"GPU (DirectML #{device_id})")
    return make


def cpu():
    return FakePose(60, nama="CPU (OpenVINO)")


def test_measure_is_median_after_warmup():
    pose = FakePose(20)
    assert ukur(pose, F, n=5) == pytest.approx(0.020)
    assert pose.calls == 6                                                 # 1 pemanasan + 5 ukur


def test_fastest_gpu_is_picked_not_device_zero():                         # Review Focus 1
    p = pilih("gpu", cpu_factory=cpu, gpu_factory=fake_factory({0: 40, 1: 9}), frame=F, lolos=True)
    assert p.jenis == "gpu" and "#1" in p.nama and p.ms == pytest.approx(9, abs=1)
    assert "#1" in p.pose.nama


def test_auto_needs_passed_equivalence():                                 # Review Focus 5
    gpus = fake_factory({0: 9})
    assert pilih("otomatis", cpu_factory=cpu, gpu_factory=gpus, frame=F, lolos=None).jenis == "cpu"
    assert pilih("otomatis", cpu_factory=cpu, gpu_factory=gpus, frame=F, lolos=False).jenis == "cpu"
    assert pilih("otomatis", cpu_factory=cpu, gpu_factory=gpus, frame=F, lolos=True).jenis == "gpu"
    forced = pilih("gpu", cpu_factory=cpu, gpu_factory=gpus, frame=F, lolos=False)
    assert forced.jenis == "gpu" and "belum lolos uji kesetaraan" in forced.pesan


def test_auto_keeps_cpu_when_gpu_is_slower():
    p = pilih("otomatis", cpu_factory=cpu, gpu_factory=fake_factory({0: 90}), frame=F, lolos=True)
    assert p.jenis == "cpu" and p.ms == pytest.approx(60, abs=1)


def test_no_gpu_means_cpu_with_reason():                                  # Review Focus 3
    p = pilih("gpu", cpu_factory=cpu, gpu_factory=None, frame=F, lolos=True)
    assert p.jenis == "cpu" and "GPU tidak tersedia" in p.pesan
    assert pilih("otomatis", cpu_factory=cpu, gpu_factory=fake_factory({}), frame=F, lolos=True).jenis == "cpu"
    failed = pilih("gpu", cpu_factory=cpu, gpu_factory=fake_factory({}), frame=F, lolos=True)
    assert failed.jenis == "cpu" and "GPU tidak tersedia" in failed.pesan


def test_cpu_mode_never_touches_gpu():
    touched = []

    def gpus(device_id):
        touched.append(device_id)
        return FakePose(5)
    p = pilih("cpu", cpu_factory=cpu, gpu_factory=gpus, frame=F, lolos=True)
    assert p.jenis == "cpu" and p.pesan == "" and touched == []


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        pilih("tpu", cpu_factory=cpu, frame=F)


def test_capacity():
    assert kapasitas(0.067) == 1 and kapasitas(0.010) == 2 and kapasitas(0.2) == 0


def test_config_fields_are_local():
    cfg = Config()
    assert cfg.hitung == "otomatis" and cfg.gpu_ms is None and cfg.gpu_id is None
    assert {"hitung", "gpu_ms", "gpu_id"} <= set(LOCAL_FIELDS)
    assert HITUNG == ("otomatis", "gpu", "cpu")


def test_equivalence_file_reader(tmp_path, monkeypatch):
    path = tmp_path / "gpu_kesetaraan.json"
    monkeypatch.setattr(ph, "KESETARAAN", path)
    assert ph.baca_kesetaraan(["X"], 416) is None
    path.write_text('{"lolos": true, "tanggal": "2026-11-20", "ringkas": "ok", "imgsz": 416, "perangkat": {"nama": ["X"]}}',
                    encoding="utf-8")
    assert ph.baca_kesetaraan(["X"], 416) is True
    path.write_text("{rusak", encoding="utf-8")
    assert ph.baca_kesetaraan(["X"], 416) is None


def test_equivalence_only_counts_for_the_tested_gpu_and_resolution(tmp_path, monkeypatch):   # Final #6
    path = tmp_path / "gpu_kesetaraan.json"
    monkeypatch.setattr(ph, "KESETARAAN", path)
    path.write_text('{"lolos": true, "imgsz": 416, "perangkat": {"nama": ["AMD Radeon(TM) Vega 8 Graphics"]}}',
                    encoding="utf-8")
    vega = ["AMD Radeon(TM) Vega 8 Graphics"]
    assert ph.baca_kesetaraan(vega, 416) is True
    assert ph.baca_kesetaraan(["AMD Radeon(TM) Graphics", "NVIDIA GeForce RTX 4060 Laptop GPU"], 416) is None
    assert ph.baca_kesetaraan(vega, 480) is None
    path.write_text('{"lolos": true}', encoding="utf-8")                  # berkas lama tanpa GPU/resolusi
    assert ph.baca_kesetaraan(vega, 416) is None


def test_gpu_runs_at_the_configured_resolution(tmp_path, monkeypatch):                    # Final #5
    from sahabat import pose_onnx
    seen = {}

    class Fake:
        def __init__(self, path, providers, provider_options, nama, imgsz):
            seen.update(imgsz=imgsz)
            self.session = type("S", (), {"get_providers": lambda s: [ph.DML, "CPUExecutionProvider"]})()
    monkeypatch.setattr(pose_onnx, "OnnxPose", Fake)
    model = tmp_path / "m.onnx"
    model.write_bytes(b"")
    monkeypatch.setattr(ph, "dml_tersedia", lambda: True)
    factory, why = ph.gpu_factory_bawaan(model, imgsz=480)
    factory(0)
    assert why == "" and seen == {"imgsz": 480}
