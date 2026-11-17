import dataclasses

from sahabat.config import Config
from sahabat.speech import Heard, HeardWord, judge

WORDS = ("anjir", "anjay", "goblok", "asu", "jancok", "kontol", "fuck", "shit")


def heard(pairs, no_speech=0.01, compression=1.1, logprob=-0.42):
    return Heard(" ".join(w for w, _ in pairs), tuple(HeardWord(w, p) for w, p in pairs), no_speech, compression, logprob)


NEAR = heard([("anjir", 0.49), ("kamu", 0.82), ("ngapain", 0.77), ("sih,", 0.89), ("desar", 0.49), ("goblok,", 0.45),
              ("asu,", 0.25), ("goblok,", 0.36), ("jancok,", 0.00), ("kontol,", 0.26), ("fuck,", 0.00), ("shit.", 0.69)])


def test_real_near_recording_gives_all_seven_words_once_each():
    v = judge([NEAR], WORDS, Config())
    assert v.reason == "" and [m.word for m in v.matches] == ["anjir", "goblok", "asu", "jancok", "kontol", "fuck", "shit"]
    assert v.matches[1].prob == 0.45 and v.text.startswith("anjir kamu")


def test_clean_sentence_gives_nothing():
    v = judge([heard([("selamat", 0.9), ("pagi", 0.9), ("kontrol", 0.8), ("asuh", 0.7)])], WORDS, Config())
    assert v.reason == "" and v.matches == ()                     # "kontrol" bukan "kontol", "asuh" bukan "asu"


def test_recited_loop_is_thrown_away():                           # model mengulang kata petunjuk
    loop = heard([("tahin", 0.2), ("di", 0.3), ("atas,", 0.2)] + [("anjay,", 0.3)] * 15, compression=1.9)
    v = judge([loop], WORDS, Config())
    assert v.reason == "berulang" and v.matches == ()


def test_word_said_twice_is_fine_three_times_in_a_row_is_a_loop():
    assert judge([heard([("asu", 0.5), ("asu", 0.5), ("kowe", 0.5)])], WORDS, Config()).reason == ""
    assert judge([heard([("asu", 0.5)] * 3)], WORDS, Config()).reason == "berulang"


def test_non_speech_squeezed_and_unsure_segments_are_dropped():
    cfg = Config()
    assert judge([heard([("anjir", 0.5)], no_speech=0.7)], WORDS, cfg).reason == "bukan_ucapan"
    assert judge([heard([("anjir", 0.5)], compression=2.5)], WORDS, cfg).reason == "mampat"
    assert judge([heard([("anjir", 0.5)], logprob=-1.6)], WORDS, cfg).reason == "tidak_yakin"
    assert all(judge([heard([("anjir", 0.5)], **kw)], WORDS, cfg).matches == ()
               for kw in ({"no_speech": 0.7}, {"compression": 2.5}, {"logprob": -1.6}))


def test_one_bad_segment_does_not_hide_a_good_one():
    v = judge([heard([("ee", 0.1)], no_speech=0.9), heard([("dasar", 0.6), ("goblok", 0.5)])], WORDS, Config())
    assert v.reason == "" and [m.word for m in v.matches] == ["goblok"]


def test_word_confidence_gate_can_be_raised():
    strict = dataclasses.replace(Config(), word_min_prob=0.4)
    assert [m.word for m in judge([NEAR], WORDS, strict).matches] == ["anjir", "goblok", "shit"]


def test_segments_without_word_times_fall_back_to_the_text():
    plain = Heard("dasar goblok kamu", (), 0.01, 1.1, -0.3)
    assert [m.word for m in judge([plain], WORDS, Config()).matches] == ["goblok"]


def test_nothing_heard_is_empty_not_an_error():
    v = judge([], WORDS, Config())
    assert v.matches == () and v.reason == "" and v.text == ""


def test_incomplete_model_folder_is_refused_before_anything_is_loaded(tmp_path):
    # Temuan tinjauan: bila tokenizer.json tidak ikut tersalin, pustakanya mencoba mengunduh saat aplikasi berjalan.
    import pytest

    from sahabat.speech import MODEL_FILES, WhisperRecognizer
    assert set(MODEL_FILES) == {"config.json", "model.bin", "tokenizer.json", "vocabulary.txt"}
    for name in MODEL_FILES:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        WhisperRecognizer(tmp_path, 2)
    with pytest.raises(FileNotFoundError):
        WhisperRecognizer(tmp_path / "tidak-ada", 2)


def test_short_encoder_removes_only_trailing_padding():
    import numpy as np
    from sahabat.speech import _trim_encoder_padding

    mel = np.zeros((80, 3000), dtype=np.float32)
    mel[:, :1051] = -1.5
    mel[:, 200:300] = 0              # sela di tengah audio tetap utuh
    compact = _trim_encoder_padding(mel, 900)
    assert compact.shape == (80, 1152)
    assert np.array_equal(compact, mel[:, :1152])
    mel[0, -1] = 1.0
    assert _trim_encoder_padding(mel, 900) is mel


# ---- dari sesi uji pemilik proyek, 15 November 2026 ----

def test_word_with_a_spoken_ending_is_announced_as_the_listed_word():
    v = judge([heard([("Kuala", 0.4), ("kontoli.", 0.3)])], WORDS, Config())      # "kontoli": i dari intonasi
    assert [m.word for m in v.matches] == ["kontol"]


def test_quiet_room_speech_with_low_confidence_is_no_longer_thrown_away():
    # "buruk pama kontol," keyakinan rata-rata sekitar -1,2: dulu dibuang "tidak_yakin" pada ambang -1,0
    v = judge([heard([("buruk", 0.2), ("pama", 0.1), ("kontol,", 0.3)], logprob=-1.2)], WORDS, Config())
    assert v.reason == "" and [m.word for m in v.matches] == ["kontol"]
    assert Config().speech_min_logprob == -1.5
    assert judge([heard([("anjir", 0.5)], logprob=-1.6)], WORDS, Config()).reason == "tidak_yakin"


def test_more_words_than_anyone_can_say_in_that_time_is_a_made_up_transcript():
    # Sesi uji kedua: potongan 1,4 detik menghasilkan 17 kata, satu di antaranya kata terdaftar yang tidak diucapkan.
    text = "kembali, ayo, ayah, ayuh, ayu, ayoh, ayi, ayur, ayol, ay, ayat, ayang, ayyuh, yuh, jancuk, ayoy, ayo"
    made_up = heard([(w, 0.3) for w in text.split()])
    words = WORDS + ("jancuk",)
    assert [m.word for m in judge([made_up], words, Config()).matches] == ["jancuk"]          # tanpa lama ucapan: lolos
    v = judge([made_up], words, Config(), seconds=1.4)
    assert v.reason == "terlalu_cepat" and v.matches == ()
    assert judge([made_up], words, Config(), seconds=6.0).reason == ""                        # 17 kata dalam 6 dtk: wajar
    assert [m.word for m in judge([NEAR], WORDS, Config(), seconds=6.0).matches][:2] == ["anjir", "goblok"]
    short = heard([("ala", 0.4), ("jancok.", 0.1)])
    assert [m.word for m in judge([short], WORDS, Config(), seconds=0.4).matches] == ["jancok"]   # ucapan pendek tetap boleh
