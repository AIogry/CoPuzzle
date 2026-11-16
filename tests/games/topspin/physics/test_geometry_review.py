"""Independent physical geometry checks for the developing 20/4 mechanism.

Static configuration sweeps are collision diagnostics, not simulated actions
or generated training trajectories. These tests do not establish successful
whole-track circulation; that still requires a contact-controlled rollout.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from copuzzle.games.topspin.physics.env import TopSpinPhysicalEnv
from copuzzle.games.topspin.physics.scene import (
    ARC_RADIUS, CENTER, HALF_STRAIGHT, TOKEN_RADIUS, TURN_CENTER, build_scene,
)


def _names(model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
            for index in range(model.ngeom)]


@pytest.fixture
def mechanism():
    model, scene = build_scene()
    data = mujoco.MjData(model)
    data.qpos[scene.arm_qpos_ids] = scene.home_qpos
    data.ctrl[scene.arm_actuator_ids] = scene.home_qpos
    mujoco.mj_forward(model, data)
    return model, data, scene


def _rotate_nominal_window(model, data, scene, angle):
    """Set one static assembly configuration for a collision measurement."""
    data.qpos[scene.turntable_qpos] = angle
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for token in range(4):
        point = TURN_CENTER[:2] + rotation @ (scene.slots[token, :2] - TURN_CENTER[:2])
        data.qpos[scene.token_qpos[token, :2]] = point - scene.slots[token, :2]
    mujoco.mj_forward(model, data)


def test_only_robot_joints_are_actuated_and_tokens_fill_the_nominal_loop(mechanism):
    model, _, scene = mechanism
    assert len(scene.token_bodies) == 20
    assert model.neq == 0, "Contact mechanism must not hide token welds or permutation constraints"
    assert set(model.actuator_trnid[:, 0].tolist()) == set(scene.arm_joint_ids.tolist())
    assert model.nu == len(scene.arm_joint_ids)
    points = scene.slots[:, :2]
    separations = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    diameter = 2 * TOKEN_RADIUS
    assert np.min(separations) >= diameter - 5e-5
    assert np.sum(separations - diameter) < diameter, \
        "Nominal loop free play must not leave enough space for an additional token"


def test_fixed_track_walls_have_physical_support_to_the_deck(mechanism):
    model, data, _ = mechanism
    names = _names(model)
    rails = [index for index, name in enumerate(names)
             if name.startswith(("arc_rail_", "lower_rail_", "turntable_surround_")) and not name.endswith("_support")]
    deck = [index for index, name in enumerate(names) if name.startswith("deck_")]
    assert rails and deck
    for rail in rails:
        support = model.geom(names[rail] + "_support").id
        assert model.geom_contype[rail] and model.geom_conaffinity[rail]
        assert model.geom_contype[support] and model.geom_conaffinity[support]
        distance = mujoco.mj_geomDistance(model, data, rail, support, 1., None)
        assert abs(distance) < 1e-8, "A floating capsule is not a supported channel wall"
        nearest_floor = min(mujoco.mj_geomDistance(model, data, support, floor, 1., None) for floor in deck)
        assert abs(nearest_floor) < 1e-8, "Channel support must reach the actual deck"


def test_complete_nominal_reversal_sweep_has_no_geometric_interference(mechanism):
    model, data, scene = mechanism
    names = _names(model)
    rotor_body = model.body("turntable").id
    moving = set(scene.token_geoms.tolist()) | set(np.flatnonzero(model.geom_bodyid == rotor_body).tolist())
    robot = set(scene.tool_geom_ids.tolist())
    for angle in np.linspace(0., np.pi, 73):
        _rotate_nominal_window(model, data, scene, angle)
        for contact in data.contact[:data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & moving and not pair & robot:
                assert contact.dist >= -5e-5, (
                    f"Nominal reversal intersects solids at {np.degrees(angle):.1f} degrees: "
                    f"{names[int(contact.geom1)]}, {names[int(contact.geom2)]}: {contact.dist} m"
                )


@pytest.mark.parametrize("side", [-1, 1])
def test_rotated_insert_physically_closes_both_inactive_track_ports(mechanism, side):
    model, data, scene = mechanism
    _rotate_nominal_window(model, data, scene, np.pi / 2)
    token = 4 if side == 1 else 19
    token_geom = int(scene.token_geoms[token])
    rotor_body = model.body("turntable").id
    # Follow the real curved centerline into the straight interface, then
    # attempt to enter a side pocket outside the rotated four-token channel.
    angles = np.linspace(np.deg2rad(75), np.pi / 2, 21)
    path = [CENTER + [side * (HALF_STRAIGHT + ARC_RADIUS * np.cos(angle)), ARC_RADIUS * np.sin(angle)]
            for angle in angles]
    path += [TURN_CENTER[:2] + [side * offset, 0]
             for offset in np.linspace(HALF_STRAIGHT, .7 * HALF_STRAIGHT, 31)]
    blocked_by_rotor = False
    for point in path:
        data.qpos[scene.token_qpos[token, :2]] = point - scene.slots[token, :2]
        mujoco.mj_forward(model, data)
        for contact in data.contact[:data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            other = second if first == token_geom else first if second == token_geom else None
            if other is not None and model.geom_bodyid[other] == rotor_body and contact.dist < -5e-5:
                blocked_by_rotor = True
    assert blocked_by_rotor, (
        "At 90 degrees an outside token can enter the rotor's side pocket: "
        "the insert needs solid banks or a passive gate, not only parallel rails"
    )


def test_gravity_is_carried_by_floor_contact_not_locked_vertical_coordinates():
    with TopSpinPhysicalEnv(max_episode_steps=100) as env:
        observation, _ = env.reset(seed=0)
        for _ in range(60):
            observation, _, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
            assert not terminated and not truncated
        names = _names(env.model)
        supported = np.zeros(20)
        for index, contact in enumerate(env.data.contact[:env.data.ncon]):
            first, second = int(contact.geom1), int(contact.geom2)
            if not any(names[geom].startswith("deck_") or names[geom] == "turntable_plate"
                       for geom in (first, second)):
                continue
            tokens = np.flatnonzero((env.scene.token_geoms == first) | (env.scene.token_geoms == second))
            if len(tokens):
                force = np.zeros(6)
                mujoco.mj_contactForce(env.model, env.data, index, force)
                supported[tokens[0]] += force[0] * abs(contact.frame[2])
        expected = env.model.body_mass[env.scene.token_bodies] * -env.model.opt.gravity[2]
        np.testing.assert_allclose(supported, expected, rtol=.02, atol=1e-4)
        for token in range(1, 21):
            joint = env.model.joint(f"token_z_{token}").id
            value = env.data.qpos[env.model.jnt_qposadr[joint]]
            lower, upper = env.model.jnt_range[joint]
            assert lower + .0005 < value < upper - .0005, "Vertical retaining limits must not carry resting weight"
        assert observation["board_valid"][0]
