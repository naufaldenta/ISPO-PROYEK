from sahabat.config import Config

from latih.ukur_aturan import clip_hits
from .helpers import ticks, with_hidden_legs
from .scenes import air_strike, handshake, hold_scene, limb_at, two


def hits(script, seconds=4.0):
    return [hit for _, hit in clip_hits(((t, script(t)) for t in ticks(seconds=seconds)), Config())]


def test_clear_air_punch_selects_nearest_target_in_three_people():
    assert [(h.kind, h.ids, h.actor) for h in hits(lambda t: air_strike(t, third=True))] == [
        ("pukulan", (1, 2), 1)]


def test_air_kick_is_detected_only_with_a_visible_leg():
    def kick(t):
        return air_strike(t, leg=True)
    assert [h.kind for h in hits(kick)] == ["tendangan"]
    assert hits(lambda t: [with_hidden_legs(kick(t)[0]), kick(t)[1]]) == []


def test_visible_arms_can_detect_air_punch_in_upper_mode():
    assert [h.kind for h in hits(lambda t: [with_hidden_legs(p) for p in air_strike(t)])] == ["pukulan"]


def test_air_punch_when_webcam_sends_seven_frames_per_second():
    frames = ((t, air_strike(t)) for t in ticks(seconds=3.0, hz=7))
    assert [h.kind for _, h in clip_hits(frames, Config())] == ["pukulan"]


def test_still_pose_handshake_and_hand_contact_do_not_signal_strike():
    def hand_contact(t):
        a, b = air_strike(t)
        return [a, limb_at(b, (355, 238), bend=20)]
    assert hits(two) == []
    assert hits(handshake) == []
    assert hits(hand_contact) == []


def test_steady_neck_and_body_holds_and_hidden_nose():
    for scene in (hold_scene, lambda t: hold_scene(t, hide_nose=True), lambda t: hold_scene(t, zone="badan")):
        assert [(h.kind, h.ids) for h in hits(scene)] == [("kuncian", (1, 2))]


def test_sustained_hold_at_four_frames_per_second():
    frames = ((t, hold_scene(t)) for t in ticks(seconds=4.0, hz=4))
    assert [h.kind for _, h in clip_hits(frames, Config())] == ["kuncian"]


def test_short_neck_touch_and_shoulder_rest_are_not_holds():
    assert hits(lambda t: hold_scene(t, seconds=1.0)) == []
    assert hits(lambda t: hold_scene(t, zone="bahu")) == []
