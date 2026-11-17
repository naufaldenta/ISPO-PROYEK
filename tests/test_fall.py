from sahabat.config import Config
from sahabat.fall import FallDetector
from sahabat.history import History

from .helpers import person, ticks

H = 480


def run_script(script, seconds):
    """Menjalankan History + FallDetector tiap 0,1 dtk; mengembalikan [(t, id)] yang dipicu."""
    hist, det = History(Config()), FallDetector(Config())
    fired = []
    for t in ticks(0.0, seconds):
        hist.update(t, script(t))
        fired += [(t, i) for i in det.tick(t, hist.tracks, H)]
    return fired


def fall_script(t, t_fall=2.0):          # tegak, lalu rebah dalam 0,4 dtk, lalu diam
    k = min(max((t - t_fall) / 0.4, 0.0), 1.0)
    return [person(1, cx=300, hip_y=300 + 80 * k, torso_deg=90 * k)]


def test_fast_fall_fires_once_within_3s():
    fired = run_script(fall_script, seconds=8.0)
    assert [i for _, i in fired] == [1]
    assert 2.0 <= fired[0][0] <= 5.0


def test_slow_lie_down_never_fires():
    slow = lambda t: [person(1, cx=300, hip_y=300 + 80 * min(t / 4, 1), torso_deg=90 * min(t / 4, 1))]
    assert run_script(slow, seconds=8.0) == []


def test_already_lying_never_fires():
    assert run_script(lambda t: [person(1, cx=300, hip_y=380, torso_deg=90)], seconds=5.0) == []


def test_refires_only_after_standing_up():
    def twice(t):
        if t < 6.0:
            return fall_script(t, 2.0)
        if t < 8.0:
            return [person(1, cx=300, hip_y=300)]
        return fall_script(t, 9.0)
    assert [i for _, i in run_script(twice, seconds=14.0)] == [1, 1]


def test_upper_mode_drop_and_vanish_fires():
    def drop(t):                         # setengah badan turun cepat lalu hilang di bawah
        if t < 2.0:
            return [person(1, cx=300, hip_y=400, visible="upper")]
        if t < 2.4:
            return [person(1, cx=300, hip_y=400 + 400 * (t - 2.0), visible="upper")]
        return []
    fired = run_script(drop, seconds=6.0)
    assert [i for _, i in fired] == [1] and fired[0][0] <= 6.0


def test_upper_mode_reappearing_cancels():
    def bob(t):
        if 2.4 <= t < 3.0:
            return []
        return [person(1, cx=300, hip_y=400 + (400 * (t - 2.0) if 2.0 <= t < 2.4 else 0), visible="upper")]
    assert run_script(bob, seconds=6.0) == []


def bottom_sliver(t, recover=False, fast=True):
    """Bahu turun keluar gambar; kepala/bahu masih terlacak, lengan sudah tidak terlihat."""
    import dataclasses
    from sahabat.types import ELBOWS, WRISTS
    if t < 2.4:
        drop = 400 * max(t - 2.0, 0) if fast else 0
        return [person(1, cx=300, hip_y=400 + drop, visible="upper")]
    if recover and t >= 2.8:
        return [person(1, cx=300, hip_y=400, visible="upper")]
    p = person(1, cx=300, hip_y=570, visible="upper")
    conf = p.conf.copy()
    conf[list(ELBOWS) + list(WRISTS)] = 0.1
    return [dataclasses.replace(p, conf=conf)]


def test_upper_drop_with_only_shoulders_remaining_at_bottom_fires_once():
    fired = run_script(bottom_sliver, seconds=10.0)
    assert len(fired) == 1 and fired[0][1] == 1
    assert 2.4 <= fired[0][0] <= 5.0


def test_upper_drop_with_bottom_sliver_then_recovery_cancels():
    assert run_script(lambda t: bottom_sliver(t, recover=True), seconds=6.0) == []


def test_bottom_sliver_without_prior_fast_drop_does_not_arm():
    assert run_script(lambda t: bottom_sliver(t, fast=False), seconds=6.0) == []


def test_upper_mode_walking_out_sideways_does_not_fire():
    walk = lambda t: [person(1, cx=300 + 200 * t, hip_y=300, visible="upper")] if t < 2.0 else []
    assert run_script(walk, seconds=5.0) == []


def test_upper_mode_standing_low_in_frame_then_leaving_does_not_fire():
    # berdiri diam dengan bahu sudah di seperlima bawah, lalu hilang: tidak ada gerak turun
    still = lambda t: [person(1, cx=300, hip_y=500, visible="upper")] if t < 2.0 else []
    assert run_script(still, seconds=5.0) == []


# ---- penilai terlatih (model tiruan: "jatuh" bila pinggul turun lebih dari 0,7 badan) ----

class StubFallModel:
    n_jobs = 8

    def __init__(self):
        self.calls, self.rows = 0, 0

    def predict_proba(self, X):
        import numpy as np
        self.calls += 1
        self.rows += len(X)
        p = (np.nan_to_num(X[:, 0], nan=0.0) > 0.7).astype(float)
        return np.column_stack([1 - p, p])


def model_detector(tmp_path):
    import joblib
    path = tmp_path / "fall_model.joblib"
    joblib.dump(StubFallModel(), path)
    return FallDetector(Config(), path)


def run_with(det, script, seconds):
    hist, fired = History(Config()), []
    for t in ticks(0.0, seconds):
        hist.update(t, script(t))
        fired += [(t, i) for i in det.tick(t, hist.tracks, H)]
    return fired


def toward_camera(t):                    # pinggul turun 1,2 badan, badan memendek, tidak pernah tampak rebah
    k = min(max((t - 2.0) / 0.4, 0.0), 1.0)
    return [person(1, cx=300, hip_y=300 + 120 * k, size=100 - 45 * k)]


def test_without_model_file_only_rules_are_used(tmp_path):
    assert FallDetector(Config()).uses_model is False
    assert FallDetector(Config(), tmp_path / "tidak-ada.joblib").uses_model is False
    assert run_script(toward_camera, seconds=8.0) == []                  # aturan sudut buta terhadap jatuh ini


def test_model_catches_a_fall_toward_the_camera(tmp_path):
    det = model_detector(tmp_path)
    assert det.uses_model and det._model.n_jobs == 1
    fired = run_with(det, toward_camera, seconds=8.0)
    assert [i for _, i in fired] == [1] and 2.0 <= fired[0][0] <= 4.0


def test_model_and_rules_together_fire_once_per_fall(tmp_path):
    fired = run_with(model_detector(tmp_path), fall_script, seconds=10.0)
    assert [i for _, i in fired] == [1]


def test_model_does_not_fire_for_sitting_down(tmp_path):
    sit = lambda t: [person(1, cx=300, hip_y=300 + 50 * min(max((t - 2.0) / 0.5, 0.0), 1.0))]
    assert run_with(model_detector(tmp_path), sit, seconds=8.0) == []


def test_model_is_called_once_per_tick_for_everyone(tmp_path):
    det = model_detector(tmp_path)
    crowd = lambda t: [person(i, cx=150 * i, hip_y=300) for i in (1, 2, 3)]
    run_with(det, crowd, seconds=3.0)
    assert det._model.calls <= 30 and det._model.rows >= 3 * 15         # tiga orang, satu panggilan per putaran


def test_person_lost_right_after_falling_still_fires(tmp_path):
    vanish = lambda t: toward_camera(t) if t < 2.6 else []                # tergeletak, lalu pelacak kehilangan dia
    fired = run_with(model_detector(tmp_path), vanish, seconds=6.0)
    assert [i for _, i in fired] == [1]


def test_second_fall_after_standing_up_fires_again(tmp_path):
    def twice(t):
        if t < 8.0:
            return toward_camera(t)
        if t < 16.0:
            return [person(1, cx=300, hip_y=300)]
        return toward_camera(t - 16.0)
    fired = run_with(model_detector(tmp_path), twice, seconds=24.0)
    assert [i for _, i in fired] == [1, 1]


def test_one_positive_tick_then_unscorable_but_still_visible_does_not_fire(tmp_path):
    def blip(t):
        if t <= 2.3:
            return toward_camera(t)                         # tepat satu langkah bertanda jatuh (t=2,3)
        if t < 3.3:
            return [person(1, cx=300, hip_y=400, size=3)]   # masih terdeteksi, tetapi badan tak terukur
        return [person(1, cx=300, hip_y=300)]
    assert run_with(model_detector(tmp_path), blip, seconds=8.0) == []


def test_one_positive_tick_then_truly_gone_still_fires(tmp_path):
    gone = lambda t: toward_camera(t) if t <= 2.3 else []   # tergeletak lalu sama sekali tak terdeteksi
    assert [i for _, i in run_with(model_detector(tmp_path), gone, seconds=6.0)] == [1]


def test_lying_person_does_not_refire_until_they_stand_up(tmp_path):
    def restless(t):
        if t < 10.0:
            return toward_camera(t)
        k = min((t - 10.0) / 0.4, 1.0)                      # masih tergeletak, bergeser turun lagi
        return [person(1, cx=300, hip_y=420 + 100 * k, size=55)]
    fired = run_with(model_detector(tmp_path), restless, seconds=18.0)
    assert [i for _, i in fired] == [1]


class WrongSizeFallModel(StubFallModel):
    n_features_in_ = 5


def test_fall_model_with_another_feature_count_is_ignored(tmp_path):
    import joblib
    path = tmp_path / "lama.joblib"
    joblib.dump(WrongSizeFallModel(), path)
    det = FallDetector(Config(), path)
    assert det.uses_model is False
    assert [i for _, i in run_with(det, fall_script, seconds=8.0)] == [1]      # aturan tetap bekerja


# ---- aturan seluruh-badan bisa dimatikan bila penilai terlatih ada ----

class NeverFallModel(StubFallModel):
    def predict_proba(self, X):
        import numpy as np
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


def never_detector(tmp_path, **overrides):
    import dataclasses

    import joblib
    path = tmp_path / "tidak-pernah.joblib"
    joblib.dump(NeverFallModel(), path)
    return FallDetector(dataclasses.replace(Config(), **overrides), path)


def test_full_body_rule_still_fires_next_to_the_model_by_default(tmp_path):
    assert [i for _, i in run_with(never_detector(tmp_path), fall_script, seconds=8.0)] == [1]


def test_full_body_rule_can_be_left_to_the_model(tmp_path):
    det = never_detector(tmp_path, fall_rule_with_model=False)
    assert run_with(det, fall_script, seconds=8.0) == []


def test_without_a_model_the_rule_always_fires(tmp_path):
    import dataclasses
    det = FallDetector(dataclasses.replace(Config(), fall_rule_with_model=False))
    assert [i for _, i in run_with(det, fall_script, seconds=8.0)] == [1]


def test_upper_body_rule_is_kept_when_the_full_body_rule_is_off(tmp_path):
    def drop(t):
        if t < 2.0:
            return [person(1, cx=300, hip_y=400, visible="upper")]
        if t < 2.4:
            return [person(1, cx=300, hip_y=400 + 400 * (t - 2.0), visible="upper")]
        return []
    det = never_detector(tmp_path, fall_rule_with_model=False)
    assert [i for _, i in run_with(det, drop, seconds=6.0)] == [1]


def test_model_votes_from_push_up_without_prior_upright_are_ignored(tmp_path):
    # Pinggul turun cukup jauh untuk model tiruan, tetapi badan sudah horizontal.
    def push_up(t):
        k = min((t % 1.0) / 0.4, 1.0)
        return [person(1, cx=300, hip_y=250 + 100 * k, torso_deg=85)]
    assert run_with(model_detector(tmp_path), push_up, seconds=6.0) == []


def test_fall_from_upright_seated_posture():
    def chair(t):
        k = min(max((t - 2) / 0.4, 0.0), 1.0)
        return [person(1, cx=300, hip_y=360 + 90 * k, torso_deg=90 * k)]
    assert [i for _, i in run_script(chair, seconds=6.0)] == [1]


def test_full_body_drop_below_image_arms_exit_alarm():
    def drop(t):
        if t < 2.5:
            k = min(max((t - 2) / 0.4, 0.0), 1.0)
            return [person(1, cx=300, hip_y=280 + 150 * k)]
        return []
    hist = History(Config())
    det = FallDetector(Config())
    events = []
    for t in ticks(seconds=6.0):
        hist.update(t, drop(t))
        events.extend((i, det.sources[i]) for i in det.tick(t, hist.tracks, H))
    assert events == [(1, "keluar_gambar")]


def test_full_body_drop_with_upper_remnant_still_alarms():
    from .helpers import with_hidden_legs

    def drop(t):
        k = min(max((t - 2) / 0.4, 0.0), 1.0)
        p = person(1, cx=300, hip_y=280 + 250 * k)
        return [p if t < 2.5 else with_hidden_legs(p)]

    assert [i for _, i in run_script(drop, seconds=6.0)] == [1]


def test_slow_approach_to_bottom_edge_does_not_trigger_fall():
    def walk(t):
        return [person(1, cx=300, hip_y=300 + 35 * t, size=100 + 10 * t)] if t < 3 else []
    assert run_script(walk, seconds=6.0) == []


def test_getting_into_push_up_position_with_hands_planted_is_not_a_fall():
    from dataclasses import replace
    from sahabat.types import ANKLES, SHOULDERS, WRISTS

    def exercise(t):
        p = fall_script(t)[0]        # torso turun dengan cepat dari posisi tegak
        if t < 2.4:
            return [p]
        kp = p.kp.copy()
        floor_y = 430
        kp[list(ANKLES), 1] = floor_y
        for shoulder, wrist in zip(SHOULDERS, WRISTS):
            kp[wrist] = (kp[shoulder, 0], floor_y)
        return [replace(p, kp=kp)]

    assert run_script(exercise, seconds=6.0) == []


# ---- turun ke bawah gambar (kamera tinggi): lengan sudah hilang saat bahu sampai di tepi bawah ----

def shoulders_drop(t, up_after=None, vanish=False, drop_s=0.5):   # 5 ukuran badan per detik
    """Tegak setengah badan, lalu bahu melesat ke tepi bawah; lengan langsung tidak terlihat."""
    import dataclasses
    from sahabat.types import ELBOWS, WRISTS
    if t < 2.0 or (up_after is not None and t >= 2.0 + up_after):
        return [person(1, cx=300, hip_y=300, visible="upper")]
    if vanish and t >= 2.0 + drop_s + 0.2:            # sempat terlihat di tepi bawah, lalu hilang
        return []
    k = min((t - 2.0) / drop_s, 1.0)
    p = person(1, cx=300, hip_y=300 + 250 * k, visible="upper")
    conf = p.conf.copy()
    conf[list(ELBOWS) + list(WRISTS)] = 0.1
    return [dataclasses.replace(p, conf=conf)]


def sources_of(script, seconds):
    hist, det = History(Config()), FallDetector(Config())
    out = []
    for t in ticks(0.0, seconds):
        hist.update(t, script(t))
        out += [(round(t, 1), det.sources[i]) for i in det.tick(t, hist.tracks, H)]
    return out


def test_fast_drop_to_the_bottom_that_stays_there_fires_once():
    fired = sources_of(shoulders_drop, seconds=10.0)
    assert len(fired) == 1 and fired[0][1] == "turun_bawah" and 5.0 <= fired[0][0] <= 6.0


def test_fast_drop_out_of_the_bottom_fires():
    fired = sources_of(lambda t: shoulders_drop(t, vanish=True), seconds=10.0)
    assert len(fired) == 1 and fired[0][0] <= 6.0


def test_squatting_near_the_camera_and_standing_up_again_does_not_fire():
    assert sources_of(lambda t: shoulders_drop(t, up_after=1.2), seconds=10.0) == []


def test_slow_descent_to_the_bottom_does_not_fire():
    assert sources_of(lambda t: shoulders_drop(t, drop_s=3.0), seconds=12.0) == []


def test_whole_body_sitting_down_on_the_floor_is_left_to_the_other_rules():
    # UR Fall adl-31..40: seluruh badan terlihat, duduk di lantai, bahu berakhir di seperlima bawah gambar
    def sit(t):
        k = min(max((t - 2.0) / 0.8, 0.0), 1.0)
        return [person(1, cx=300, hip_y=300 + 160 * k, size=75)]
    assert "turun_bawah" not in [source for _, source in sources_of(sit, seconds=10.0)]


def test_shoulders_already_low_that_jitter_down_do_not_arm():
    import dataclasses
    from sahabat.types import ELBOWS, WRISTS

    def seated_near_camera(t):           # duduk dekat kamera tinggi, bahu sudah di bawah; dua gambar meleset turun
        jitter = {30: 30, 31: 60}.get(round(t * 10), 0)
        p = person(1, cx=300, hip_y=470 + jitter, size=50, visible="upper")
        if not jitter:
            return [p]
        conf = p.conf.copy()
        conf[list(ELBOWS) + list(WRISTS)] = 0.1     # lengan hilang saat meleset: aturan setengah badan tidak siaga
        return [dataclasses.replace(p, conf=conf)]
    assert sources_of(seated_near_camera, seconds=10.0) == []
