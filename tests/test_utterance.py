import numpy as np

from sahabat.config import Config
from sahabat.utterance import Segmenter

RATE = 16000


def tone(seconds, db=-20.0, hz=300.0):
    t = np.arange(int(seconds * RATE)) / RATE
    return (32767 * 10 ** (db / 20) * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def voice(seconds, db=-20.0):
    """Nada yang kekerasannya naik-turun 4 kali per detik, seperti suku kata. `tone` yang rata dipakai untuk dengung."""
    t = np.arange(int(seconds * RATE)) / RATE
    return (tone(seconds, db).astype(np.float64) * (0.55 + 0.45 * np.sin(2 * np.pi * 4 * t))).astype(np.int16)


def hiss(seconds, db=-65.0, seed=0):
    x = np.random.default_rng(seed).standard_normal(int(seconds * RATE))
    return (32767 * 10 ** (db / 20) * x).astype(np.int16)


def run(audio, block=1600):
    seg, out = Segmenter(Config()), []
    for i in range(0, len(audio), block):
        out += seg.feed(audio[i:i + block])
    return out + seg.flush()


def seconds(utterances):
    return [round(len(u) / RATE, 1) for u in utterances]


def test_one_word_between_silences_is_one_utterance():
    out = run(np.concatenate([hiss(1.0), voice(1.0), hiss(1.5)]))
    assert len(out) == 1 and 1.0 <= len(out[0]) / RATE <= 1.9      # ucapan + awalan + hening penutup
    assert out[0].dtype == np.int16


def test_click_shorter_than_min_is_ignored():
    assert run(np.concatenate([hiss(1.0), voice(0.1), hiss(1.5)])) == []


def test_two_words_split_by_a_long_pause_are_two():
    assert len(run(np.concatenate([hiss(1.0), voice(0.8), hiss(1.0), voice(0.8), hiss(1.0)]))) == 2


def test_short_pause_inside_a_sentence_does_not_split():
    assert len(run(np.concatenate([hiss(1.0), voice(0.8), hiss(0.2), voice(0.8), hiss(1.0)]))) == 1


def test_isolated_word_finishes_early_and_longer_speech_keeps_its_pause():
    for duration, expected in ((0.60, 1), (1.50, 0)):
        seg = Segmenter(Config())
        pcm = np.concatenate([hiss(0.60), tone(duration), hiss(0.39)])
        assert len(seg.feed(pcm)) == expected
        assert len(seg.feed(hiss(0.15))) == 1 - expected


def test_long_speech_is_cut_at_max_with_overlap():
    out = run(np.concatenate([hiss(1.0), voice(14.0), hiss(1.0)]))
    assert len(out) == 3 and all(s <= 6.1 for s in seconds(out))
    assert sum(len(u) for u in out) >= 14.0 * RATE + 2 * 0.4 * RATE  # potongan bertumpuk ±0,5 dtk


def test_steady_hum_is_not_speech():
    assert run(tone(8.0, db=-40.0, hz=100.0)) == []                 # dengung tetap: lantai bising ikut naik


def test_speech_over_a_hum_is_still_found():
    hum = tone(4.0, db=-45.0, hz=100.0).astype(np.int32)
    hum[RATE * 2:RATE * 3] += voice(1.0)
    assert len(run(hum.astype(np.int16))) == 1


def test_hum_that_starts_later_is_learned_as_noise_then_speech_is_found_again():
    # kipas menyala di detik ke-2: potongan paksa pertama rata, jadi dibuang dan lantai bising dinaikkan
    audio = np.concatenate([hiss(2.0), tone(20.0, db=-35.0, hz=100.0)]).astype(np.int32)
    audio[RATE * 15:RATE * 16] += voice(1.0, db=-12.0)
    out = run(audio.astype(np.int16))
    assert len(out) == 1 and 1.0 <= len(out[0]) / RATE <= 1.9


def test_block_size_does_not_change_the_result():
    audio = np.concatenate([hiss(1.0), voice(1.0), hiss(1.0), voice(0.7), hiss(1.0)])
    assert seconds(run(audio, 1600)) == seconds(run(audio, 480)) == seconds(run(audio, 7000))


def test_flush_gives_an_unfinished_utterance():
    seg = Segmenter(Config())
    assert seg.feed(np.concatenate([hiss(1.0), voice(1.0)])) == []
    assert len(seg.flush()) == 1 and seg.flush() == []


def test_speech_just_under_the_limit_then_silence_is_one_utterance_without_a_tail():
    # Temuan tinjauan: setelah potongan paksa, sisa 0,1 dtk suara + hening dulu keluar sebagai ucapan kedua
    # yang isinya hampir seluruhnya suara yang sudah dikirim (kata di akhir kalimat terdengar dua kali).
    assert seconds(run(np.concatenate([hiss(1.0), voice(5.9), hiss(2.0)]))) == [6.0]


def wandering_noise(seconds, db=-44.0, swing_db=3.5, seed=3):
    """Bising ruangan yang kekerasannya naik-turun pelan (bukan dengung rata): orang jauh, TV, jalan."""
    t = np.arange(int(seconds * RATE)) / RATE
    x = np.random.default_rng(seed).standard_normal(len(t))
    gain = 10 ** ((db + swing_db * np.sin(2 * np.pi * 0.3 * t)) / 20)
    return (32767 * gain * x).astype(np.int16)


def test_room_noise_after_a_quiet_start_does_not_keep_everything_inside_one_utterance():
    # Sesi uji pemilik proyek: lantai bising terpelajari dari awal yang senyap lalu membeku, sehingga bising ruangan
    # terus dianggap suara dan SEMUA potongan keluar 6 detik, memotong kata di sembarang tempat.
    audio = np.concatenate([hiss(0.5, db=-62.0), wandering_noise(30.0)]).astype(np.int32)
    for at in (8.0, 19.0):                                           # dua kalimat 1,5 dtk, hanya ±13 dB di atas bising
        audio[int(at * RATE):int((at + 1.5) * RATE)] += voice(1.5, db=-28.0)
    seg, found, fed = Segmenter(Config()), [], 0
    pcm = audio.astype(np.int16)
    for i in range(0, len(pcm) // 480 * 480, 480):
        fed = i + 480
        found += [((fed - len(u)) / RATE, len(u) / RATE) for u in seg.feed(pcm[i:fed])]
    assert all(dur < 5.9 for _, dur in found), found                 # tidak ada potongan paksa
    # Puncak bising yang naik-turun 7 dB masih lolos sesekali (gerbang 6 dB), tetapi tidak lagi seluruh aliran.
    assert sum(dur for _, dur in found) < 0.8 * len(pcm) / RATE
    for at in (8.0, 19.0):                                           # kedua kalimat utuh di dalam satu ucapan
        assert any(start <= at + 0.1 and start + dur >= at + 1.4 for start, dur in found), (at, found)
