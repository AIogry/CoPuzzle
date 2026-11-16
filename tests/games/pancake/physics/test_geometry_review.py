"""Independent scratch-IK collision checks for the full robot and apparatus.

Held discs follow rigid relative transforms only in this static planning
diagnostic. This does not modify a live environment or prove grasp stability;
actual contact rollouts provide that separate evidence.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from copuzzle.games.pancake.physics.control import PoseIK
from copuzzle.games.pancake.physics.scene import (
    BOSS_HALF, JAW_TRAVEL, N, PITCH, STACK_XY, SUPPORT_TOP, build_scene,
)
from copuzzle.games.pancake.physics.scripted import PancakeFlipPolicy


def _slerp(start, end, fraction):
    dot = float(np.dot(start, end))
    if dot < -1e-12:
        end, dot = -end, -dot
    if dot > .999999:
        result = (1 - fraction) * start + fraction * end
    else:
        angle = np.arccos(np.clip(dot, -1, 1))
        result = (np.sin((1 - fraction) * angle) * start + np.sin(fraction * angle) * end) / np.sin(angle)
    return result / np.linalg.norm(result)


@pytest.mark.parametrize("prefix", [1, 2, 3, 4])
def test_scripted_pose_path_avoids_robot_world_and_self_collisions(prefix):
    model, scene = build_scene()
    scratch = mujoco.MjData(model)
    controller = PoseIK(model, scene)
    scratch.qpos[scene.arm_qpos_ids] = scene.home_qpos
    for token in range(N):
        address = scene.token_qpos_ids[token]
        scratch.qpos[address:address + 3] = [*STACK_XY, SUPPORT_TOP + BOSS_HALF + (N - token - 1) * PITCH]
        scratch.qpos[address + 3:address + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, scratch)
    position, quaternion, grip = scene.home_tool_position.copy(), scene.tool_down_quat.copy(), 0.
    scratch.qpos[scene.arm_qpos_ids] = controller.solve(scratch, position, quaternion)
    mujoco.mj_forward(model, scratch)
    token_bodies = set(scene.token_body_ids.tolist())
    robot_bodies = set(range(1, model.nbody)) - token_bodies
    held_geoms = set(scene.token_geom_ids[:prefix].ravel().tolist())
    tool_geoms = set(scene.tool_geom_ids.tolist())
    local_positions = local_quaternions = None

    for phase, target, target_quaternion, target_grip, _ in PancakeFlipPolicy(prefix).waypoints:
        start, start_quaternion, start_grip = position.copy(), quaternion.copy(), grip
        angle = 2 * np.arccos(np.clip(abs(np.dot(start_quaternion, target_quaternion)), -1, 1))
        count = max(1, int(np.ceil(np.linalg.norm(target - start) / .002)), int(np.ceil(angle / .014)))
        if phase == "lift":
            transform = scratch.site_xmat[scene.tool_site_id].reshape(3, 3)
            origin = scratch.site_xpos[scene.tool_site_id]
            local_positions = (scratch.xpos[scene.token_body_ids[:prefix]] - origin) @ transform
            actual = np.empty(4)
            mujoco.mju_mat2Quat(actual, transform.ravel())
            inverse = np.empty(4)
            mujoco.mju_negQuat(inverse, actual)
            local_quaternions = []
            for body in scene.token_body_ids[:prefix]:
                local = np.empty(4)
                mujoco.mju_mulQuat(local, inverse, scratch.xquat[body])
                local_quaternions.append(local)
        previous_joints = scratch.qpos[scene.arm_qpos_ids].copy()
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            position = (1 - fraction) * start + fraction * target
            quaternion = _slerp(start_quaternion, target_quaternion, fraction)
            grip = (1 - fraction) * start_grip + fraction * target_grip
            joints = controller.solve(scratch, position, quaternion)
            scratch.qpos[scene.arm_qpos_ids] = joints
            scratch.qpos[scene.jaw_qpos_id] = grip * JAW_TRAVEL
            mujoco.mj_forward(model, scratch)
            assert controller.last_position_error < 1e-4, phase
            assert controller.last_rotation_error < 2e-4, phase
            assert np.max(np.abs(joints - previous_joints)) < .05, f"Discontinuous IK branch in {phase}"
            previous_joints = joints.copy()
            if local_positions is not None and phase not in ("release", "withdraw"):
                transform = scratch.site_xmat[scene.tool_site_id].reshape(3, 3)
                origin = scratch.site_xpos[scene.tool_site_id]
                actual = np.empty(4)
                mujoco.mju_mat2Quat(actual, transform.ravel())
                for token, local in enumerate(local_positions):
                    address = scene.token_qpos_ids[token]
                    scratch.qpos[address:address + 3] = origin + transform @ local
                    orientation = np.empty(4)
                    mujoco.mju_mulQuat(orientation, actual, local_quaternions[token])
                    scratch.qpos[address + 3:address + 7] = orientation
                mujoco.mj_forward(model, scratch)
            for contact in scratch.contact[:scratch.ncon]:
                first, second = int(contact.geom1), int(contact.geom2)
                if contact.dist >= -5e-5:
                    continue
                if not (model.geom_bodyid[first] in robot_bodies or model.geom_bodyid[second] in robot_bodies):
                    continue
                # Fork/disc grasp contact needs a dynamic force audit; its
                # compliant compression is intentionally not judged by IK.
                if ((first in tool_geoms and second in held_geoms)
                        or (second in tool_geoms and first in held_geoms)):
                    continue
                bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
                          for g in (first, second)]
                pytest.fail(f"Whole-robot collision during {phase}: {bodies}, penetration {-contact.dist} m")

    # Unlike the intentional grasp contacts above, the final retreat must
    # physically clear every disc, including the largest lower-stack disc.
    physical_tool = [g for g in tool_geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
    clearance = min(mujoco.mj_geomDistance(model, scratch, int(tool), int(disc), 1., None)
                    for tool in physical_tool for disc in scene.token_geom_ids.ravel())
    assert clearance > .002, "Retreated tool still overlaps or closely grazes the stack"
