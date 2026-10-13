"""Switch timing independent of MuJoCo contact generation."""
import numpy as np
import pytest

from copuzzle.games.lights_out.physics.events import ButtonEventDetector


def test_requires_real_contact_at_press_depth():
    detector = ButtonEventDetector(2)
    assert detector.update([0.009, 0.0], [False, True]).tolist() == []
    assert detector.armed.tolist() == [1, 1]
    assert detector.update([0.009, 0.0], [True, True]).tolist() == [0]


def test_hold_has_one_event_and_release_rearms():
    detector = ButtonEventDetector(1)
    counts = 0
    for depth in [0.0, 0.004, 0.008, 0.010, 0.009, 0.008, 0.005, 0.004]:
        counts += len(detector.update([depth], [True]))
    assert counts == 1
    assert detector.armed.tolist() == [0]
    assert detector.update([0.003], [False]).tolist() == []
    assert detector.armed.tolist() == [1]
    assert detector.update([0.008], [True]).tolist() == [0]


def test_substep_sampling_catches_complete_press_and_release_between_endpoints():
    detector = ButtonEventDetector(1)
    depths = [0.0, 0.004, 0.009, 0.006, 0.002, 0.0]
    fired = [(step, button) for step, depth in enumerate(depths)
             for button in detector.update([depth], [depth > 0])]
    assert fired == [(2, 0)]
    assert depths[0] == depths[-1]


def test_two_presses_keep_count_even_when_parity_cancels():
    detector = ButtonEventDetector(2)
    counts = np.zeros(2, dtype=np.int32)
    for depths in ([0, 0], [.009, .009], [.003, .010], [.009, .010]):
        counts[detector.update(depths, [True, True])] += 1
    np.testing.assert_array_equal(counts, [2, 1])
    np.testing.assert_array_equal(counts % 2, [0, 1])


def test_reset_does_not_fire_or_alias_state():
    detector = ButtonEventDetector(2)
    detector.reset([0.009, 0.0])
    assert detector.armed.tolist() == [0, 1]
    assert detector.update([0.009, 0.009], [True, True]).tolist() == [1]
    exposed = detector.armed
    exposed[:] = 1
    assert detector.armed.tolist() == [0, 0]
    detector.reset()
    assert detector.armed.tolist() == [1, 1]


@pytest.mark.parametrize('count', [0, -1, True, 1.5, None])
def test_invalid_button_count(count):
    with pytest.raises(ValueError):
        ButtonEventDetector(count)


@pytest.mark.parametrize('press,release', [(0, 0), (.003, .008), (.008, -.001),
                                          (np.nan, 0), (.008, np.inf), (True, 0)])
def test_invalid_thresholds(press, release):
    with pytest.raises(ValueError):
        ButtonEventDetector(1, press_depth=press, release_depth=release)


@pytest.mark.parametrize('depths,contacts', [([0], [True, False]), ([0, np.nan], [0, 0]),
                                           ([0, 0], [0, 2]), ([[0, 0]], [0, 0]),
                                           ([True, False], [True, False])])
def test_invalid_substep_input(depths, contacts):
    with pytest.raises(ValueError):
        ButtonEventDetector(2).update(depths, contacts)
