"""Physical facility acceptance: support, bore, full travel and visible identity.

These checks inspect compiled MuJoCo geometry and actual contacts. Static
qpos sweeps are geometric diagnostics, never generated motion trajectories.
The historical v1 profile is kept only to reproduce its existing artifacts.
"""

import os

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from copuzzle.games.lights_out.physics.env import PhysicalPuzzleEnv
from copuzzle.games.lights_out.physics.scene import BUTTON_TRAVEL, build_scene
from copuzzle.games.lights_out.rules import grid_rule


def _names(model, prefix):
    return [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(prefix)]


def _vertical_bounds(model, data, geom):
    """Exact for the vertical cylinders used for the stem, cap and base."""
    assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER
    rotation = data.geom_xmat[geom].reshape(3, 3)
    np.testing.assert_allclose(np.abs(rotation[:, 2]), [0, 0, 1], atol=1e-12)
    return data.geom_xpos[geom, 2] + np.array([-1, 1]) * model.geom_size[geom, 1]


def _distance(model, data, first, second):
    return mujoco.mj_geomDistance(model, data, int(first), int(second), 1., None)


def _unsolved_reset(env):
    start = np.zeros(env.rule.n_buttons, dtype=np.uint8)
    return env.reset(options={"state": start, "goal": env.rule.transition(start, env.rule.n_buttons - 1)})


@pytest.fixture(scope="module")
def compiled_facility():
    model, scene = build_scene(grid_rule(2, 3))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, scene


def test_caps_are_solid_supported_and_guided(compiled_facility):
    model, data, scene = compiled_facility
    assert scene.metadata["facility_profile"] == "button_board_v2"
    for index, joint in enumerate(scene.button_joint_ids):
        cap = model.geom(f"button_cap_{index}").id
        stem = model.geom(f"button_stem_{index}").id
        sleeve = _names(model, f"button_sleeve_{index}_")
        assert sleeve, "Each moving stem needs a fixed hollow guide"
        assert model.geom_type[cap] == mujoco.mjtGeom.mjGEOM_CYLINDER
        assert model.geom_type[stem] == mujoco.mjtGeom.mjGEOM_CYLINDER
        assert model.geom_bodyid[cap] == model.geom_bodyid[stem] == model.jnt_bodyid[joint]
        assert model.geom_rgba[cap, 3] == 1, "The contact surface must be opaque and closed"
        assert model.geom_contype[cap] and model.geom_conaffinity[cap]
        assert model.geom_contype[stem] and model.geom_conaffinity[stem]
        assert all(model.geom_contype[g] and model.geom_conaffinity[g] for g in sleeve)
        assert all(model.geom_bodyid[g] == 0 for g in sleeve), "The sleeve must stay fixed to the grounded facility"
        np.testing.assert_allclose(data.geom_xpos[cap, :2], data.geom_xpos[stem, :2], atol=1e-12)
        assert _vertical_bounds(model, data, stem)[1] == pytest.approx(_vertical_bounds(model, data, cap)[0], abs=1e-10)
        assert model.body_mass[model.geom_bodyid[cap]] == pytest.approx(.02)
        assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE
        np.testing.assert_allclose(model.jnt_axis[joint], [0, 0, -1])
        np.testing.assert_allclose(model.jnt_range[joint], [0, BUTTON_TRAVEL])
        assert model.jnt_stiffness[joint] > 0


def test_sleeves_have_a_real_open_bore_and_solid_collision_walls(compiled_facility):
    model, data, scene = compiled_facility
    data.qpos[scene.button_qpos_ids] = 0
    mujoco.mj_forward(model, data)
    for index, rest in enumerate(scene.button_rest_positions):
        sleeve = set(_names(model, f"button_sleeve_{index}_"))
        assert sleeve, "Missing support housing"
        moving_body = model.body(f"button_{index}").id
        hit = np.zeros(1, dtype=np.int32)
        # Excluding cap/stem exposes the bore. A solid 'sleeve' cylinder fails:
        # this ray must reach its support floor, not a fictitious closed top.
        origin = np.array([rest[0], rest[1], .10])
        distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]),
                                 None, True, moving_body, hit)
        assert hit[0] == model.geom(f"button_base_{index}").id
        assert origin[2] - distance == pytest.approx(.032, abs=1e-8)
        for angle in (np.arange(12) + .31) * 2 * np.pi / 12:
            point = origin + [.0175 * np.cos(angle), .0175 * np.sin(angle), 0]
            distance = mujoco.mj_ray(model, data, point, np.array([0., 0., -1.]),
                                     None, True, moving_body, hit)
            assert hit[0] in sleeve, "Housing must surround the bore without radial openings"
            assert point[2] - distance == pytest.approx(.054, abs=1e-7)


def test_stationary_housing_is_supported_continuously_by_base_and_panel(compiled_facility):
    model, data, scene = compiled_facility
    panel, ground = model.geom("button_panel").id, model.geom("floor").id
    assert abs(_distance(model, data, panel, ground)) <= 1e-8
    for index in range(len(scene.button_qpos_ids)):
        base = model.geom(f"button_base_{index}").id
        assert abs(_distance(model, data, base, panel)) <= 1e-8
        sleeve = _names(model, f"button_sleeve_{index}_")
        assert sleeve
        assert all(abs(_distance(model, data, geom, base)) <= 1e-8 for geom in sleeve), \
            "The fixed housing must rest on its support, not float above the floor"


@pytest.mark.parametrize("depth", [0., .003, .006, .009, .012])
def test_complete_stroke_has_connected_support_and_no_solid_intersections(compiled_facility, depth):
    model, data, scene = compiled_facility
    data.qpos[scene.button_qpos_ids] = depth
    mujoco.mj_forward(model, data)
    panel = model.geom("button_panel").id
    for index in range(len(scene.button_qpos_ids)):
        cap = model.geom(f"button_cap_{index}").id
        stem = model.geom(f"button_stem_{index}").id
        floor = model.geom(f"button_base_{index}").id
        sleeve = _names(model, f"button_sleeve_{index}_")
        stem_low, stem_high = _vertical_bounds(model, data, stem)
        cap_low, _ = _vertical_bounds(model, data, cap)
        np.testing.assert_allclose(stem_high, cap_low, atol=1e-10)
        assert stem_low < .054 and stem_high > .034, "The stem must remain captured inside the guide across the full stroke"
        assert _distance(model, data, stem, floor) >= .0015
        assert _distance(model, data, stem, panel) >= .0015
        assert _distance(model, data, cap, floor) >= .0015
        assert _distance(model, data, cap, panel) >= .0015
        assert all(_distance(model, data, stem, geom) >= .0015 for geom in sleeve)
        assert all(_distance(model, data, cap, geom) >= .0015 for geom in sleeve)


def test_gravity_equilibrium_is_below_release_without_phantom_presses():
    with PhysicalPuzzleEnv(grid_rule(2, 3), max_episode_steps=150) as env:
        observation, _ = _unsolved_reset(env)
        start = observation["board"].copy()
        for _ in range(80):
            observation, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
            assert not terminated and not truncated
            assert not info["event_log"]
            assert not env._contacts().any()
        masses = env.model.body_mass[env.model.jnt_bodyid[env.scene.button_joint_ids]]
        stiffness = env.model.jnt_stiffness[env.scene.button_joint_ids]
        predicted = masses * -env.model.opt.gravity[2] / stiffness
        np.testing.assert_allclose(observation["button_displacements"], predicted, atol=5e-5)
        assert np.max(observation["button_displacements"]) < env._events.release_depth
        assert observation["button_armed"].all()
        np.testing.assert_array_equal(observation["board"], start)


def test_external_compression_without_probe_contact_does_not_toggle():
    with PhysicalPuzzleEnv(grid_rule(2, 3), max_episode_steps=100) as env:
        observation, _ = _unsolved_reset(env)
        start = observation["board"].copy()
        # Apply a physical generalized force, never an intended button action.
        env.data.qfrc_applied[env.scene.button_dof_ids[0]] = 2.
        compressed = False
        for _ in range(30):
            observation, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
            compressed |= observation["button_displacements"][0] >= env._events.press_depth
            assert not terminated and not truncated and not info["event_log"]
            assert not env._contacts().any()
        assert compressed, "The test force must actually cross the press threshold"
        np.testing.assert_array_equal(observation["board"], start)
        env.data.qfrc_applied[:] = 0
        for _ in range(30):
            observation, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
            assert not info["event_log"]
        assert observation["button_displacements"][0] <= env._events.release_depth


def test_real_probe_press_hold_release_and_second_press_with_clear_housing():
    with PhysicalPuzzleEnv(grid_rule(2, 3), max_episode_steps=250) as env:
        observation, _ = _unsolved_reset(env)
        initial = observation["board"].copy()
        rest = observation["button_rest_positions"][0]
        high, down = rest + [0, 0, .10], rest - [0, 0, .010]
        seen = []
        for target, count, expected_events in ((high, 30, 0), (down, 40, 1),
                                                (down, 25, 0), (high, 35, 0), (down, 40, 1)):
            events = []
            for _ in range(count):
                command = np.clip((target - observation["end_effector_position"]) / env.action_scale, -1, 1).astype(np.float32)
                observation, _, terminated, truncated, info = env.step(command)
                assert not terminated and not truncated
                events.extend(info["event_log"])
                # A cap housing collision would invalidate the nominal stroke.
                for contact in env.data.contact[:env.data.ncon]:
                    names = {mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
                             for g in (contact.geom1, contact.geom2)}
                    if "button_cap_0" in names and any(name.startswith("button_sleeve_0_") for name in names):
                        assert contact.dist >= -1e-4
            assert len(events) == expected_events
            assert all(event["button"] == 0 for event in events)
            seen.extend(events)
            if target is high:
                assert observation["button_armed"][0]
            else:
                assert env._contacts()[0] and not observation["button_armed"][0]
        assert len(seen) == 2
        np.testing.assert_array_equal(observation["board"], initial)


def test_render_identity_uses_actual_caps_hides_debug_sites_and_not_goal_bits():
    if os.environ.get("MUJOCO_GL") not in {"egl", "osmesa"}:
        pytest.skip("Set MUJOCO_GL=egl or osmesa for exact render segmentation checks")
    rule = grid_rule(2, 3)
    with PhysicalPuzzleEnv(rule, render_mode="rgb_array") as env:
        start = np.zeros(rule.n_buttons, dtype=np.uint8)
        env.reset(options={"state": start, "goal": rule.transition(start, 5)})
        # Even if a user exposes model sites, ordinary environment rendering
        # must not leak attachment/tool/debug markers into RGB observations.
        env.model.site_group[:] = 0
        env.model.site_rgba[:, 3] = 1
        before = env.render()
        scene = env._renderer.scene
        assert not any(geom.objtype == mujoco.mjtObj.mjOBJ_SITE for geom in scene.geoms[:scene.ngeom])
        env._renderer.enable_segmentation_rendering()
        segmentation = env._renderer.render().copy()
        env._renderer.disable_segmentation_rendering()
        for cap in env.scene.button_geom_ids:
            pixels = (segmentation[:, :, 0] == cap) & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
            assert pixels.sum() >= 5, "Each actual button contact cap must appear in the RGB scene"
        assert not np.any(segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_SITE)
        env.reset(options={"state": start, "goal": rule.transition(start, 4)})
        np.testing.assert_array_equal(env.render(), before)
        colors_before = env.model.geom_rgba[env.scene.button_geom_ids].copy()
        changed = rule.transition(start, 0)
        env.reset(options={"state": changed, "goal": rule.transition(start, 5)})
        colors_after = env.model.geom_rgba[env.scene.button_geom_ids]
        np.testing.assert_array_equal(np.any(colors_before != colors_after, axis=1), changed != start)


def test_legacy_geometry_keeps_its_historical_identity_for_replay_only():
    model, scene = build_scene(grid_rule(2, 3), facility_profile="legacy_v1")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert scene.metadata["scene_version"] == "copuzzle.ur5e_buttons.v1"
    assert scene.model_sha256 == "f4c13ec691392e3acfc9238c03a0cac0d39f51838eb7cdc63f2d69c6151dde87"
    assert not _names(model, "button_stem_")
    cap, base = model.geom("button_cap_0").id, model.geom("button_base_0").id
    gap = _vertical_bounds(model, data, cap)[0] - _vertical_bounds(model, data, base)[1]
    assert gap == pytest.approx(.018), "Legacy replay must not silently reinterpret the previous flawed facility"
