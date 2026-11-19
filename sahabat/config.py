"""Semua angka yang bisa disetel ada di sini, plus baca/tulis berkas pengaturan.

Ambang adalah bagian dari program dan hanya diubah di berkas ini. Berkas pengaturan di data/ hanya
menyimpan pilihan milik satu laptop (resolusi hasil autotune, port, kamera), supaya folder aplikasi
yang diperbarui tidak tertimpa ambang lama.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
MODELS_DIR = APP_DIR / "models"
SETTINGS_PATH = DATA_DIR / "pengaturan.json"
WORDS_PATH = APP_DIR / "kata-kasar.txt"       # daftar kata kasar; disunting sekolah
VIDEO_DIR = APP_DIR / "video"                 # video cadangan pengganti kamera (dipilih dari layar)
KAMERA_PATH = DATA_DIR / "kamera.json"            # daftar kamera CCTV (kata sandi terenkripsi)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".m4v")
_SETTINGS_LOCK = threading.RLock()

LOCAL_FIELDS = ("imgsz", "tuned_for", "slow", "port", "camera", "words_enabled", "microphone",
                "camera_flip", "clips_enabled", "zone", "lan_enabled", "lan_code", "cctv_enabled")


@dataclass
class Config:
    # laju dan kecepatan
    rate_hz: float = 10.0
    fps_target: float = 10.0
    fps_floor: float = 8.0
    cctv_enabled: bool = False        # Mode CCTV menyala saat SAHABAT dimulai (disimpan dari layar)
    cctv_hold_s: float = 5.0          # Mode CCTV: kamera tetap penuh selama ini sesudah orang terakhir terlihat
    cctv_safety_s: float = 5.0        # kamera siaga tetap dinilai pose sekali tiap selang ini
    cctv_motion_hz: float = 5.0       # laju cek gerak murah kamera siaga
    imgsz_candidates: tuple[int, ...] = (480, 416, 320)
    imgsz: int | None = None          # hasil autotune
    tuned_for: str | None = None      # platform.node() saat autotune
    slow: bool = False                # hasil autotune: di bawah fps_floor
    # orang dan mode pandang
    max_people: int = 4
    kp_min_conf: float = 0.5
    mode_hysteresis_s: float = 0.5
    history_s: float = 2.0
    track_ttl_s: float = 1.0
    # Saat jatuh cepat YOLO sering kehilangan orangnya sebentar dan pelacak memberinya nomor baru.
    # Nomor baru yang muncul dekat orang yang baru saja hilang meneruskan riwayat orang itu.
    stitch_s: float = 1.0             # paling lama hilang
    stitch_dist: float = 2.0          # paling jauh, dalam ukuran badan
    ghost_dist: float = 0.12          # dua deteksi yang titik tubuhnya sedekat ini (per ukuran kotak) = satu orang
    # jatuh, seluruh badan
    fall_upright_deg: float = 30.0
    fall_lying_deg: float = 60.0
    fall_hip_speed: float = 1.0       # ukuran badan per detik, ke bawah
    fall_drop_window_s: float = 1.0
    fall_stay_s: float = 1.0
    # jatuh, setengah badan
    upper_drop_speed: float = 1.0
    upper_bottom_frac: float = 0.8    # seperlima bawah gambar
    upper_vanish_s: float = 1.0
    fall_need_upright: bool = True
    fall_upright_before_s: float = 2.0
    fall_edge_frac: float = 0.95
    fall_edge_vanish_s: float = 1.0
    # jatuh, penilai terlatih (dipakai bila models/fall_model.joblib ada; aturan di atas tetap jalan)
    fall_window_s: float = 2.0
    # 0,9 dipilih dari sapuan ambang pada NTU + video sungguhan: alarm keliru turun,
    # jatuh yang tertangkap nyaris tidak berubah.
    fall_model_threshold: float = 0.9
    fall_confirm_k: int = 3
    fall_confirm_n: int = 5
    # False: bila penilai terlatih ada, aturan seluruh-badan di atas tidak ikut memicu (aturan setengah badan tetap).
    fall_rule_with_model: bool = True
    # None membaca ambang dukungan aturan seluruh badan dari model yang terpasang.
    fall_rule_model_threshold: float | None = None
    # kekerasan
    # 4 ukuran badan (kira-kira 2 m). Dengan 2,5 pasangan di video sungguhan hanya dinilai sekitar satu detik
    # saat aksinya terjadi, terlalu singkat untuk konfirmasi.
    pair_max_dist: float = 4.0
    violence_window_s: float = 1.0
    # Titik kerja dari validasi silang NTU RGB+D + UT-Interaction per klip:
    # menaikkan ambang mengurangi alarm keliru tetapi lebih banyak kejadian terlewat.
    violence_threshold: float = 0.8
    confirm_k: int = 6
    confirm_n: int = 10
    pair_cooldown_s: float = 5.0
    rule_limb_speed: float = 2.0
    rule_limb_dist: float = 0.6
    rule_push_speed: float = 1.5
    violence_min_motion: float = 1.0    # ukuran badan per detik sebelum peluang model bisa memicu
    gesture_enabled: bool = True
    strike_window_s: float = 0.6
    strike_speed: float = 3.0
    strike_reach: float = 0.8
    strike_straight: float = 0.8
    strike_reach_gain: float = 0.5
    strike_hand_clear: float = 0.3
    strike_confirm: int = 1
    strike_confirm_s: float = 3.0
    hold_s: float = 1.5
    hold_gap_s: float = 0.2
    hold_neck_dist: float = 0.28
    hold_center_frac: float = 0.22
    hold_straight: float = 0.8
    lost_s: float = 1.0
    lost_votes: int = 3
    # kata kasar lewat mikrofon
    words_enabled: bool = True              # tombol hidup/mati di layar; pilihan terakhir diingat per laptop
    microphone: int | str | None = None     # None = perangkat masukan bawaan Windows
    utt_silence_s: float = 0.5              # hening sepanjang ini mengakhiri satu ucapan
    utt_short_silence_s: float = 0.36       # akhir lebih cepat untuk satu ucapan singkat
    utt_min_s: float = 0.3
    utt_max_s: float = 6.0                  # ucapan lebih panjang dipotong paksa
    utt_overlap_s: float = 0.5              # tumpang-tindih antar potongan paksa
    # Bersuara = sekian dB di atas lantai bising. Lantai = persentil rendah kekerasan semua bingkai dalam jendela
    # terakhir. 6 dB dipilih dari dua sesi uji pemilik proyek (suara hanya 10-13 dB di atas bising ruangan):
    # dengan 12 dB sebagian besar ucapan pelan hilang, dengan 8 dB muncul satu peringatan keliru.
    utt_gate_db: float = 6.0
    utt_floor_s: float = 10.0
    utt_floor_pct: float = 10.0
    utt_abs_db: float = -55.0               # ... dan tidak lebih pelan dari ini
    speech_threads: int = 2
    speech_encoder_min_s: float = 9.0      # bantalan minimum encoder untuk potongan pendek; 30 = jalur lama
    # Pencegah pengulangan di pengenal (lihat sahabat/speech.py). Dasarnya baru dua rekaman satu orang, jadi bisa
    # dimatikan untuk dibandingkan: python -m latih.suara ... --set speech_no_repeat_ngram=0 speech_repetition_penalty=1
    speech_no_repeat_ngram: int = 3
    speech_repetition_penalty: float = 1.1
    speech_max_no_speech: float = 0.6       # segmen yang menurut pengenal kemungkinan bukan ucapan dibuang
    speech_max_compression: float = 2.4     # teks yang sangat mampat = pengulangan
    # Keyakinan rata-rata satu segmen. -1,0 terlalu ketat untuk suara orang di ruangan biasa: pada sesi uji pemilik
    # proyek (suara ±10 dB di atas bising ruangan) 7 dari 23 ucapan dibuang, termasuk satu yang memuat kata kasar.
    speech_min_logprob: float = -1.5
    # Gerbang keyakinan per kata dimatikan (0): terukur 15 Nov, kata yang benar terdengar pun nilainya rendah
    # (jancok 0,00; fuck 0,00; asu 0,25; anjir 0,49). Naikkan hanya setelah diukur pada rekaman.
    word_min_prob: float = 0.0
    word_repeat_max: int = 3                # kata yang sama sekian kali berturut-turut = model mengulang petunjuk
    # Lebih banyak kata daripada yang bisa diucapkan orang dalam waktu itu = teks karangan pengenal (sesi uji kedua:
    # 17 kata dari potongan 1,4 detik, salah satunya kata terdaftar). Batas = sekian kata per detik + speech_words_slack.
    speech_max_words_per_s: float = 6.0
    speech_words_slack: int = 3
    words_prompt_limit: int = 40
    words_notice_s: float = 6.0             # lama pita kuning tampil
    words_max_shown: int = 3
    # aplikasi
    port: int = 8765
    camera: int | str | None = None   # None = pilih otomatis
    camera_flip: bool = False         # balik kiri-kanan untuk driver yang mengirim gambar tercermin
    # riwayat kejadian dan klip bukti (data/riwayat/), hanya di laptop ini
    clips_enabled: bool = True        # tombol di panel Riwayat; pilihan terakhir diingat per laptop
    clip_before_s: float = 8.0        # klip mulai sekian detik sebelum kejadian (inspeksi: cukup konteks)
    clip_after_s: float = 5.0         # ... dan berakhir sekian detik sesudahnya
    clip_keep_days: float = 7.0       # catatan dan klip dihapus otomatis setelah sekian hari
    # zona pantau: titik [x, y] ternormalisasi (0..1); None = seluruh gambar. Digambar di layar, diingat per laptop.
    zone: list | None = None
    # layar dari HP / laptop lain di WiFi yang sama (sahabat/jaringan.py); mati secara bawaan
    lan_enabled: bool = False
    lan_port: int = 8770
    lan_code: str | None = None       # kode akses; dibuat sekali lalu diingat per laptop


def load(path: Path) -> Config:
    """Membaca pilihan laptop ini; berkas hilang atau rusak menghasilkan nilai bawaan."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Config()
    if not isinstance(raw, dict):
        return Config()
    try:
        return dataclasses.replace(Config(), **{k: raw[k] for k in LOCAL_FIELDS if k in raw})
    except TypeError:
        return Config()


def save(cfg: Config, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _SETTINGS_LOCK:
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
                tmp = f.name
                json.dump({k: getattr(cfg, k) for k in LOCAL_FIELDS}, f, indent=2)
            os.replace(tmp, path)
        finally:
            if tmp is not None and os.path.exists(tmp):
                os.unlink(tmp)


def update_local(path: Path, **changes) -> None:
    """Gabungkan pilihan perangkat/tombol agar request bersamaan tidak menimpa pilihan lain."""
    if not set(changes) <= set(LOCAL_FIELDS):
        raise ValueError("Hanya pilihan laptop yang boleh disimpan")
    with _SETTINGS_LOCK:
        save(dataclasses.replace(load(path), **changes), path)
