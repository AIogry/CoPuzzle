"""Review metrics distinguish token engagement from a correct final board."""

import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
EXAMPLES = ROOT / "examples"
FIXTURES = Path(__file__).parent / "fixtures"
spec = importlib.util.spec_from_file_location("contact_engagement_analysis", EXAMPLES / "contact_engagement_review.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_center_relative_motion_separates_carried_contact_and_sliding():
    rows = [{"action_index": index, "post_contacts": [1]} for index in range(4)]
    translation = np.array([0., .001, .003, .006, .010])
    tool = np.column_stack((translation, np.zeros(5), np.zeros(5)))
    tokens = tool[:, None, :].copy()
    carried = review.relative_motion({"end_effector_position": tool, "token_positions": tokens}, rows, 1, -1, .05)
    assert carried["tool_tangential_displacement_m"] == pytest.approx(.010)
    assert carried["relative_tangential_total_variation_m"] == pytest.approx(0.)
    tokens[:, 0, 0] *= .2
    sliding = review.relative_motion({"end_effector_position": tool, "token_positions": tokens}, rows, 1, -1, .05)
    assert sliding["relative_tangential_net_displacement_m"] == pytest.approx(.008)
    assert sliding["relative_tangential_total_variation_m"] == pytest.approx(.008)
    # Every sampled contact ID can still be the target while large slip occurs.
    assert sliding["reported_contact_fraction"] == 1.


def test_contact_segments_preserve_dual_contact_handoff_and_dropouts():
    contacts = [[8], [8], [2, 8], [2], [], [2]]
    rows = [{"action_index": index + 10, "post_contacts": ids} for index, ids in enumerate(contacts)]
    segments = review.contact_segments(rows, .05)
    assert [item["reported_token_ids"] for item in segments] == [[8], [2, 8], [2], [], [2]]
    assert [item["steps"] for item in segments] == [2, 1, 1, 1, 1]
    assert segments[0]["first_action_index"] == 10
    assert segments[0]["last_action_index"] == 11


def test_archived_mixed_trace_exposes_actual_selected_token_handoffs():
    archive = ROOT / "artifacts" / "topspin-contact-v1" / "mixed.npz"
    if not archive.exists():
        pytest.skip("Historical review artifact is not installed")
    report = review.audit_topspin(archive, FIXTURES / "topspin_legacy_policy.py")
    assert report["action_reconstruction_max_absolute_error"] == 0.
    right, left = report["operations"]
    assert right["selected_token_id"] == left["selected_token_id"] == 8
    assert right["reported_push_contact_ids"] == [1, 2, 8]
    assert left["reported_push_contact_ids"] == [8, 9, 19]
    assert right["exclusive_other_contact_steps"] == 50
    assert left["exclusive_other_contact_steps"] == 102
    assert right["metrics"]["relative_tangential_net_displacement_m"] == pytest.approx(.04106981653998221)
    assert left["metrics"]["relative_tangential_net_displacement_m"] == pytest.approx(.06869913237192626)
    assert "not continuous" in report["measurement_limits"][0]


def test_mismatched_policy_snapshot_is_rejected_before_executing_it(tmp_path):
    archive = ROOT / "artifacts" / "topspin-contact-v1" / "mixed.npz"
    if not archive.exists():
        pytest.skip("Historical review artifact is not installed")
    source = tmp_path / "unmatched.py"
    source.write_text("raise RuntimeError('This unverified source must not execute')\n")
    with pytest.raises(ValueError, match="checksum"):
        review.audit_topspin(archive, source)


@pytest.fixture
def pin_socket_model():
    mujoco = pytest.importorskip("mujoco")
    from copuzzle.games.topspin.physics.scene import build_scene
    model, scene = build_scene(facility_profile="engaged_v2")
    data = mujoco.MjData(model)
    data.qpos[scene.arm_qpos_ids] = scene.home_qpos
    mujoco.mj_forward(model, data)
    return mujoco, model, data, scene


def test_every_socket_has_a_real_open_entrance_and_a_colliding_blind_floor(pin_socket_model):
    mujoco, model, data, scene = pin_socket_model
    from copuzzle.games.topspin.physics.scene import SOCKET_BOTTOM
    hit = np.zeros(1, dtype=np.int32)
    for token, body in enumerate(scene.token_bodies):
        center = data.xpos[body]
        origin = center + [0, 0, .025]
        distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]), None, True, -1, hit)
        assert hit[0] == scene.token_geoms[token], "The socket entrance was sealed by a full cylinder or convex hull"
        assert origin[2] - distance == pytest.approx(center[2] + SOCKET_BOTTOM, abs=1e-8)
        assert model.geom_contype[hit[0]] and model.geom_conaffinity[hit[0]]


def test_socket_contact_identity_covers_every_real_wall_without_hidden_actuation(pin_socket_model):
    _, model, _, scene = pin_socket_model
    actual = {}
    for token_id, body in enumerate(scene.token_bodies, start=1):
        for geom in np.flatnonzero(model.geom_bodyid == body):
            if model.geom_contype[geom] or model.geom_conaffinity[geom]:
                actual[int(geom)] = token_id
        assert model.body_mass[body] == pytest.approx(.012)
    assert scene.geom_token_ids == actual
    assert all(sum(owner == token for owner in actual.values()) > 1 for token in range(1, 21))
    assert model.neq == 0
    assert model.nu == len(scene.arm_joint_ids)
    assert set(model.actuator_trnid[:, 0]) == set(scene.arm_joint_ids)


def test_inserted_pin_has_clearance_and_transmits_lateral_contact_only_to_its_socket(pin_socket_model):
    mujoco, model, data, scene = pin_socket_model
    from copuzzle.games.lights_out.physics.control import DampedIK
    from copuzzle.games.topspin.physics.scene import PIN_RADIUS, SOCKET_RADIUS
    controller = DampedIK(model, scene)
    selected = 12
    center = data.xpos[scene.token_bodies[selected - 1]].copy()
    pin, shaft = model.geom("probe_tip").id, model.geom("probe_shaft").id
    floor = int(scene.token_geoms[selected - 1])
    walls = [geom for geom, token in scene.geom_token_ids.items() if token == selected and geom != floor]
    assert model.geom_type[pin] == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert model.geom_size[pin, 0] == pytest.approx(PIN_RADIUS)
    target = center + [0, 0, -.002]
    data.qpos[scene.arm_qpos_ids] = controller.solve(data, target)
    mujoco.mj_forward(model, data)
    assert controller.last_position_error < 1e-4
    assert mujoco.mj_geomDistance(model, data, pin, floor, 1., None) > .0005
    # The convex-sector socket is polygonal, so its inscribed clearance is
    # slightly smaller than the nominal radius difference.
    assert min(mujoco.mj_geomDistance(model, data, pin, wall, 1., None) for wall in walls) > .7 * (SOCKET_RADIUS - PIN_RADIUS)
    assert min(mujoco.mj_geomDistance(model, data, shaft, geom, 1., None) for geom in [floor, *walls]) > .0005
    for angle in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
        target = center + [(SOCKET_RADIUS - PIN_RADIUS + .0003) * np.cos(angle),
                           (SOCKET_RADIUS - PIN_RADIUS + .0003) * np.sin(angle), -.002]
        data.qpos[scene.arm_qpos_ids] = controller.solve(data, target)
        mujoco.mj_forward(model, data)
        assert any(mujoco.mj_geomDistance(model, data, pin, wall, 1., None) < -.0001 for wall in walls), \
            "A displaced pin must encounter an actual colliding socket wall"
        assert all(mujoco.mj_geomDistance(model, data, pin, geom, 1., None) >= -1e-5
                   for geom, token in scene.geom_token_ids.items() if token != selected)


def test_sequence_keeps_executing_failure_withdrawal_until_driver_finishes():
    from copuzzle.games.topspin.physics.scripted import SequencePolicy

    class WithdrawingDriver:
        """Model the public driver protocol independently of its implementation."""
        failure = "socket engagement lost"
        completed = False
        finished = False
        calls = 0

        def __call__(self, observation):
            self.calls += 1
            self.finished = bool(observation["tool_clear"][0])
            return np.array([0., 0., 0. if self.finished else 1.], dtype=np.float32)

    sequence = SequencePolicy(["shift_right"])
    driver = WithdrawingDriver()
    sequence.driver = driver
    for _ in range(3):
        command = sequence({"tool_clear": np.array([0], dtype=np.uint8)})
        assert command[2] > 0
        assert not sequence.finished and sequence.failure is None
    assert driver.calls == 3
    sequence({"tool_clear": np.array([1], dtype=np.uint8)})
    # The wrapper may forward completion on the same or the next call, but
    # it must never stop before the driver has observed actual clearance.
    sequence({"tool_clear": np.array([1], dtype=np.uint8)})
    assert sequence.finished and not sequence.completed
    assert sequence.failure == "socket engagement lost"


def test_frozen_v2_mixed_trace_preserves_identity_and_verifies_every_full_permutation():
    archive = ROOT / "artifacts" / "topspin-engaged-v2" / "mixed.npz"
    if not archive.exists():
        pytest.skip("Final engaged review artifact is not installed")
    report = review.audit_engaged_topspin(archive)
    assert not report["event_record_errors"]
    assert report["actual_event_sequence_matches_intention"]
    assert report["final_full_20_token_board_matches_composition"]
    assert report["final_tool_clear"] and report["final_board_valid"]
    assert report["classified_event_count"] == report["final_atomic_operation_count"] == 4
    for operation in report["operation_reviews"]:
        assert operation["wrong_positive_load_identity_steps"] == 0
        assert operation["max_radial_excess_over_nominal_clearance_m"] < .0002
        assert operation["selected_force_above_0_05N_steps"] > 0
        # The review must retain observed unload samples instead of turning
        # a correct final board into a claim of continuous no-slip contact.
        assert operation["unloaded_contact_steps"] > 0


def test_final_goal_does_not_hide_an_incorrect_intermediate_event(tmp_path):
    archive = ROOT / "artifacts" / "topspin-engaged-v2" / "mixed.npz"
    if not archive.exists():
        pytest.skip("Final engaged review artifact is not installed")
    copy = tmp_path / "mixed.npz"
    shutil.copyfile(archive, copy)
    header = json.loads(archive.with_suffix(".json").read_text())
    event = next(info["event_log"][0] for info in header["infos"] if info["event_log"])
    event["state_after"][4], event["state_after"][5] = event["state_after"][5], event["state_after"][4]
    copy.with_suffix(".json").write_text(json.dumps(header))
    report = review.audit_engaged_topspin(copy)
    assert report["final_full_20_token_board_matches_composition"]
    assert report["event_record_errors"]
    assert not report["event_record_errors"][0]["full_20_token_transformation_matches"]
