"""Independent asset, kinematics and real actuator/contact checks."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from copuzzle.games.lights_out.physics.control import DampedIK
from copuzzle.games.lights_out.physics.scene import ASSET_DIRECTORY, build_scene
from copuzzle.games.lights_out.presets import TASK_PRESETS
from copuzzle.games.lights_out.rules import get_rule, graph_rule


def initialized(task):
    model, scene = build_scene(get_rule(task))
    data = mujoco.MjData(model)
    data.qpos[scene.arm_qpos_ids] = scene.home_qpos
    mujoco.mj_forward(model, data)
    controller = DampedIK(model, scene)
    data.qpos[scene.arm_qpos_ids] = controller.solve(data, scene.home_tool_position)
    data.ctrl[scene.arm_actuator_ids] = data.qpos[scene.arm_qpos_ids]
    mujoco.mj_forward(model, data)
    return model, scene, data, controller


def test_pinned_upstream_asset_manifest_matches_every_packaged_file():
    manifest = json.loads((ASSET_DIRECTORY / "provenance.json").read_text())
    assert manifest["source_repository"] == "https://github.com/google-deepmind/mujoco_menagerie"
    assert manifest["source_commit"] == "c6c202003c72e87d079db69160e499fb0d83c535"
    assert manifest["license"] == "BSD-3-Clause"
    assert "Copyright 2018 ROS Industrial Consortium" in (ASSET_DIRECTORY / "LICENSE").read_text()
    assert {"LICENSE", "README.md", "ur5e.xml"} <= manifest["files"].keys()
    assert len([name for name in manifest["files"] if name.startswith("assets/")]) == 20
    for name, record in manifest["files"].items():
        content = (ASSET_DIRECTORY / name).read_bytes()
        assert len(content) == record["bytes"]
        assert hashlib.sha256(content).hexdigest() == record["sha256"]
        assert manifest["source_commit"] in record["source_url"]


@pytest.mark.parametrize("task", [preset.task_id for preset in TASK_PRESETS])
def test_all_standard_buttons_have_reachable_hover_and_press_poses(task):
    model, scene, data, controller = initialized(task)
    assert model.nu == 6
    assert model.nq == model.nv == 6 + len(scene.button_joint_ids)
    assert len(scene.arm_joint_ids) == len(scene.home_qpos) == 6
    np.testing.assert_allclose(data.site_xpos[scene.tool_site_id], scene.home_tool_position, atol=5e-5)
    scratch = mujoco.MjData(model)
    for position in scene.button_rest_positions:
        for height in (0.16, position[2] - 0.012):
            target = position.copy()
            target[2] = height
            solution = controller.solve(data, target)
            assert controller.last_position_error < 5e-5
            assert controller.last_rotation_error < 5e-4
            assert np.all(solution >= model.jnt_range[scene.arm_joint_ids, 0])
            assert np.all(solution <= model.jnt_range[scene.arm_joint_ids, 1])
            scratch.qpos[scene.arm_qpos_ids] = solution
            mujoco.mj_forward(model, scratch)
            np.testing.assert_allclose(scratch.site_xpos[scene.tool_site_id], target, atol=5e-5)
            # The site's positive z axis really points down, not just its position.
            rotation = scratch.site_xmat[scene.tool_site_id].reshape(3, 3)
            np.testing.assert_allclose(rotation[:, 2], [0, 0, -1], atol=5e-4)


def test_ik_uses_scratch_state_without_teleporting_live_simulation():
    model, scene, data, controller = initialized("grid-3x3")
    before = {name: getattr(data, name).copy() for name in ("qpos", "qvel", "ctrl", "site_xpos")}
    time_before = data.time
    result = controller.solve(data, scene.button_rest_positions[0])
    assert result.shape == (6,)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(data, name), value)
    assert data.time == time_before
    result[:] = 0
    repeated = controller.solve(data, scene.button_rest_positions[0])
    assert np.any(repeated)


def test_zero_cartesian_action_holds_arm_and_springs_do_not_self_trigger():
    model, scene, data, controller = initialized("grid-3x3")
    for _ in range(100):
        data.ctrl[scene.arm_actuator_ids] = controller.solve(data, data.site_xpos[scene.tool_site_id].copy())
        mujoco.mj_step(model, data, nstep=25)
        mujoco.mj_forward(model, data)
        assert np.max(data.qpos[scene.button_qpos_ids]) < 0.003
        assert np.isfinite(data.qpos).all()
    np.testing.assert_allclose(data.site_xpos[scene.tool_site_id], scene.home_tool_position, atol=1e-4)
    # Actual gravity/spring equilibrium: mg/k, not a button pinned by resetting qpos.
    np.testing.assert_allclose(data.qpos[scene.button_qpos_ids], 0.02 * 9.81 / 200, atol=1e-6)
    assert scene.metadata["arm_gravity_compensation"] is True
    assert scene.metadata["button_gravity_compensation"] is False


def drive(model, scene, data, controller, target, *, steps=65):
    touched, depressed = set(), set()
    max_depth = np.zeros(len(scene.button_joint_ids))
    for _ in range(steps):
        current = data.site_xpos[scene.tool_site_id].copy()
        data.ctrl[scene.arm_actuator_ids] = controller.solve(
            data, current + np.clip(target - current, -0.02, 0.02))
        for _ in range(25):
            mujoco.mj_step(model, data)
            depths = data.qpos[scene.button_qpos_ids]
            max_depth = np.maximum(max_depth, depths)
            depressed.update(np.flatnonzero(depths >= 0.008).tolist())
            for contact in data.contact:
                if contact.dist > 0:
                    continue
                for probe, cap in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                    if probe in scene.tool_geom_ids and cap in scene.button_geom_ids:
                        touched.add(int(np.flatnonzero(scene.button_geom_ids == cap)[0]))
        mujoco.mj_forward(model, data)
        if np.max(np.abs(target - data.site_xpos[scene.tool_site_id])) < 0.0008:
            break
    assert np.isfinite(data.qpos).all()
    return touched, depressed, max_depth


@pytest.mark.parametrize("task", ["grid-3x3", "grid-4x4", "grid-4x5", "grid-4x6", "graph-path24"])
def test_every_button_can_be_physically_pressed_and_released(task):
    model, scene, data, controller = initialized(task)
    for button, position in enumerate(scene.button_rest_positions):
        target = position.copy()
        target[2] = 0.16
        _, unexpected, _ = drive(model, scene, data, controller, target)
        assert not unexpected
        target[2] = position[2] - 0.012
        touched, depressed, max_depth = drive(model, scene, data, controller, target)
        assert touched == depressed == {button}
        assert max_depth[button] >= 0.008
        assert max_depth[button] < 0.013
        target[2] = 0.16
        drive(model, scene, data, controller, target)
        assert np.max(data.qpos[scene.button_qpos_ids]) < 0.003


def test_graph_layout_is_independent_of_edges_and_optional_display_coordinates():
    first = graph_rule(np.zeros((9, 9), dtype=np.uint8), name="empty")
    second = get_rule("graph-cycle9")
    third = graph_rule(second.adjacency, name="display", coordinates=np.ones((9, 2)) * 99)
    model_a, scene_a = build_scene(first)
    _, scene_b = build_scene(second)
    _, scene_c = build_scene(third)
    np.testing.assert_array_equal(scene_a.button_rest_positions, scene_b.button_rest_positions)
    assert scene_a.model_sha256 == scene_b.model_sha256 == scene_c.model_sha256
    assert len(scene_a.model_sha256) == 64
    assert scene_a.button_rest_positions.flags.writeable is False
    with pytest.raises(ValueError):
        scene_a.button_rest_positions[0, 0] = 100
    json.dumps(scene_a.metadata, allow_nan=False)


def test_non_square_grid_maps_row_to_x_column_to_y():
    _, scene = build_scene(get_rule("grid-4x5"))
    positions = scene.button_rest_positions
    np.testing.assert_allclose(positions[1] - positions[0], [0, 0.065, 0])
    np.testing.assert_allclose(positions[5] - positions[0], [0.065, 0, 0])
    np.testing.assert_allclose(positions[7] - positions[0], [0.065, 0.13, 0])


@pytest.mark.parametrize("physics_dt", [0, -0.002, 1.0, float("nan"), float("inf"), True, "0.002", 0.002j])
def test_invalid_physics_timestep_rejected(physics_dt):
    with pytest.raises(ValueError):
        build_scene(get_rule("grid-3x3"), physics_dt=physics_dt)


@pytest.mark.parametrize("target", [[1, 2], [0.5, 0, float("nan")], [0.5, 0, float("inf")],
                                    ["0.5", "0", "0.2"], [0.5j, 0, 0.2]])
def test_invalid_ik_target_rejected(target):
    _, _, data, controller = initialized("grid-3x3")
    with pytest.raises(ValueError):
        controller.solve(data, target)


def test_scene_fingerprint_tracks_simulation_timestep():
    _, first = build_scene(get_rule("grid-3x3"), physics_dt=0.002)
    _, second = build_scene(get_rule("grid-3x3"), physics_dt=0.001)
    assert first.model_sha256 != second.model_sha256
