import numpy as np

from sahabat.config import Config
from sahabat.penjadwal import GerakMurah, KamInfo, Penjadwal


def k(id, **kw):
    base = dict(prioritas=False, tersambung=True, penuh=False, last_step=0.0, gerak=False, fps=0.0, penuh_sejak=None)
    return KamInfo(id=id, **{**base, **kw})


def test_defaults_come_from_config():
    cfg = Config()
    assert (cfg.cctv_hold_s, cfg.cctv_safety_s, cfg.cctv_motion_hz) == (5.0, 5.0, 5.0)


def test_full_cameras_due_every_tenth_second_priority_first():
    p = Penjadwal()
    cams = [k("a", penuh=True, last_step=0.95), k("b", penuh=True, last_step=0.85, prioritas=True),
            k("c", penuh=True, last_step=0.80)]
    assert p.pilih(1.0, cams) == ["b", "c"]                  # a baru 0,05 dtk lalu; prioritas didahulukan


def test_idle_cameras_only_on_motion_or_safety():
    p = Penjadwal()
    assert p.pilih(10.0, [k("a", last_step=9.0)]) == []
    assert p.pilih(10.0, [k("a", last_step=9.0, gerak=True)]) == ["a"]
    assert p.pilih(10.0, [k("a", last_step=4.9)]) == ["a"]   # pengaman 5 dtk
    assert p.pilih(10.0, [k("a", tersambung=False, last_step=0.0)]) == []


def test_full_before_idle():
    p = Penjadwal()
    assert p.pilih(10.0, [k("idle", gerak=True, last_step=9.7), k("busy", penuh=True, last_step=9.8)]) == ["busy", "idle"]


def test_status_slow_only_after_two_seconds_below_floor():
    p = Penjadwal()
    assert p.status(k("a", penuh=True, fps=5.0, penuh_sejak=9.0, last_step=9.95), 10.0) == "penuh"   # baru 1 dtk
    assert p.status(k("a", penuh=True, fps=5.0, penuh_sejak=7.0, last_step=9.95), 10.0) == "lambat"
    assert p.status(k("a", penuh=True, fps=9.5, penuh_sejak=0.0, last_step=9.95), 10.0) == "penuh"
    assert p.status(k("a", penuh=True, fps=9.5, penuh_sejak=9.9, last_step=9.6), 10.0) == "lambat"   # tidak dilayani
    assert p.status(k("a"), 10.0) == "siaga" and p.status(k("a", tersambung=False), 10.0) == "terputus"


def test_motion_check_rate():
    p = Penjadwal()
    assert p.perlu_cek_gerak(1.0, 0.75) and not p.perlu_cek_gerak(1.0, 0.85)


def test_cheap_motion():
    g = GerakMurah()
    still = np.full((360, 640, 3), 90, np.uint8)
    assert g.berubah(still) and not g.berubah(still.copy())
    moved = still.copy()
    moved[100:300, 200:400] = 200
    assert g.berubah(moved)
    assert not g.berubah(moved.copy())


def test_idle_camera_is_never_starved_by_busy_ones():
    """Kamera siaga yang bergerak sudah ≥ 0,5 dtk menunggu, atau jatuh tempo pengaman 5 dtk, didahulukan dari kamera
    penuh biasa (spec: naik ke penuh ≤ 0,5 dtk; tidak ada kamera yang terlewat terus-menerus)."""
    p = Penjadwal()
    busy = [k("a", penuh=True, last_step=9.85), k("b", penuh=True, last_step=9.88)]
    assert p.pilih(10.0, busy + [k("idle", gerak=True, last_step=float("-inf"))])[0] == "idle"
    assert p.pilih(10.0, busy + [k("idle", gerak=True, last_step=9.7)])[-1] == "idle"     # baru 0,3 dtk
    assert p.pilih(10.0, busy + [k("idle", last_step=4.0)])[0] == "idle"                    # pengaman 5 dtk
    vip = k("vip", penuh=True, prioritas=True, last_step=9.8)
    assert p.pilih(10.0, busy + [vip, k("idle", gerak=True, last_step=0.0)])[:2] == ["idle", "vip"]
