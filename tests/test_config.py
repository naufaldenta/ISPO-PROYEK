from sahabat.config import LOCAL_FIELDS, Config, load, save


def test_defaults_match_spec():
    c = Config()
    assert (c.rate_hz, c.fps_target, c.fps_floor) == (10.0, 10.0, 8.0)
    assert c.imgsz_candidates == (480, 416, 320)
    assert (c.max_people, c.mode_hysteresis_s, c.pair_max_dist) == (4, 0.5, 4.0)       # spec: 2,5; dilonggarkan setelah uji video
    assert (c.pair_cooldown_s, c.violence_window_s) == (5.0, 1.0)


def test_roundtrip(tmp_path):
    p = tmp_path / "pengaturan.json"
    save(Config(imgsz=416, tuned_for="LAPTOP-A"), p)
    assert load(p).imgsz == 416 and load(p).tuned_for == "LAPTOP-A"


def test_update_local_preserves_other_device_choices(tmp_path):
    from sahabat.config import update_local
    p = tmp_path / "p.json"
    save(Config(imgsz=416), p)
    update_local(p, camera="camera:usb", microphone="mic:usb")
    update_local(p, words_enabled=False)
    cfg = load(p)
    assert (cfg.camera, cfg.microphone, cfg.words_enabled, cfg.imgsz) == ("camera:usb", "mic:usb", False, 416)


def test_roundtrip_keeps_tuple_and_equality(tmp_path):
    p = tmp_path / "sub" / "pengaturan.json"
    save(Config(), p)
    assert load(p) == Config()


def test_missing_or_corrupt_file_gives_defaults(tmp_path):
    assert load(tmp_path / "tidak-ada.json") == Config()
    bad = tmp_path / "rusak.json"
    bad.write_text("{bukan json", encoding="utf-8")
    assert load(bad) == Config()


def test_unknown_keys_are_ignored(tmp_path):
    p = tmp_path / "p.json"
    p.write_text('{"imgsz": 320, "kunci_lama": 1}', encoding="utf-8")
    assert load(p).imgsz == 320


def test_violence_defaults_are_the_chosen_operating_point():
    c = Config()
    assert (c.violence_threshold, c.confirm_k, c.confirm_n) == (0.8, 6, 10)     # dipilih dari NTU + video sungguhan


def test_settings_file_holds_only_this_laptops_choices(tmp_path):
    # ambang adalah bagian dari program; berkas pengaturan lama tidak boleh menimpa ambang versi baru
    import json
    p = tmp_path / "p.json"
    save(Config(imgsz=416, tuned_for="X", slow=True, port=9000, camera=1, fall_stay_s=9.0), p)
    assert set(json.loads(p.read_text(encoding="utf-8"))) == {"imgsz", "tuned_for", "slow", "port", "camera",
                                                                 "words_enabled", "microphone", "camera_flip"}
    got = load(p)
    assert (got.imgsz, got.tuned_for, got.slow, got.port, got.camera) == (416, "X", True, 9000, 1)
    assert got.fall_stay_s == Config().fall_stay_s
    p.write_text(json.dumps({"imgsz": 320, "violence_threshold": 0.1}), encoding="utf-8")
    assert load(p).violence_threshold == Config().violence_threshold


def test_word_settings_defaults_and_local_fields():
    c = Config()
    assert (c.words_enabled, c.microphone, c.words_notice_s, c.words_max_shown) == (True, None, 6.0, 3)
    assert (c.utt_silence_s, c.utt_min_s, c.utt_max_s, c.word_min_prob, c.word_repeat_max) == (0.5, 0.3, 6.0, 0.0, 3)
    assert {"words_enabled", "microphone"} <= set(LOCAL_FIELDS)


def test_decoder_anti_repetition_can_be_switched_for_comparison():
    c = Config()
    assert (c.speech_no_repeat_ngram, c.speech_repetition_penalty) == (3, 1.1)
