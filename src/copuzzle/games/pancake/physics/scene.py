"""UR5e with a visible contact-only fork clamp and gravity-supported discs.

The symmetric central bosses are physical spacers, leaving a 4 mm peripheral
fork insertion gap. These are rigid laboratory sorting discs, not deformable
food. No weld, adhesion, or invisible token carrier is present.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from copuzzle.games.lights_out.physics.scene import _robot_assets

N = 4
STACK_XY = np.array([.52, 0.])
SUPPORT_TOP = .28
PITCH = .014
DISC_HALF = .005
BOSS_HALF = .007
RADII = (.023, .029, .035, .041)
TOOL_X, TOOL_Z = .25, .12
JAW_OPEN_GAP = .10
JAW_TRAVEL = .098
FORK_FORWARD = .025
WITHDRAW_DISTANCE = max(RADII) + FORK_FORWARD + .003


def numbers(value):
    return ' '.join(format(float(item), '.16g') for item in value)


@dataclass
class PancakeScene:
    arm_joint_ids: np.ndarray
    arm_qpos_ids: np.ndarray
    arm_dof_ids: np.ndarray
    arm_actuator_ids: np.ndarray
    tool_site_id: int
    tool_down_quat: np.ndarray
    workspace_bounds: np.ndarray
    home_qpos: np.ndarray
    home_tool_position: np.ndarray
    jaw_joint_id: int
    jaw_qpos_id: int
    jaw_dof_id: int
    jaw_actuator_id: int
    token_body_ids: np.ndarray
    token_qpos_ids: np.ndarray
    token_dof_ids: np.ndarray
    tool_geom_ids: np.ndarray
    token_geom_ids: np.ndarray
    model_sha256: str
    metadata: dict


def _digits(body, number, radius):
    """Small raised seven-segment numerals on both faces; no collision or mass."""
    segments = {
        'a': (0, .010, .006, .0008), 'b': (.006, .005, .0008, .005),
        'c': (.006, -.005, .0008, .005), 'd': (0, -.010, .006, .0008),
        'e': (-.006, -.005, .0008, .005), 'f': (-.006, .005, .0008, .005),
        'g': (0, 0, .006, .0008),
    }
    selected = {1: 'bc', 2: 'abged', 3: 'abgcd', 4: 'fgbc'}[number]
    for side in (-1, 1):
        for segment in selected:
            x, y, sx, sy = segments[segment]
            # Offset from the small central spacer so each numeral is readable.
            x += radius * .40
            ET.SubElement(body, 'geom', name=f'label_{number}_{side}_{segment}', type='box',
                          pos=numbers([x, y, side * (DISC_HALF + .0002)]),
                          size=numbers([sx, sy, .00015]), rgba='.18 .075 .025 1',
                          contype='0', conaffinity='0', mass='0', group='2')


def build_scene(*, physics_dt=.001):
    xml, assets, provenance = _robot_assets()
    root = ET.fromstring(xml)
    root.set('model', 'copuzzle_pancake_contact_clamp_v1')
    for key in root.findall('keyframe'):
        root.remove(key)
    for body in root.iter('body'):
        body.set('gravcomp', '1')
    root.find('option').attrib.update(timestep=str(physics_dt), gravity='0 0 -9.81',
                                    integrator='implicitfast', cone='elliptic',
                                    iterations='100', tolerance='1e-10')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='720')
    ET.SubElement(visual, 'headlight', ambient='.4 .4 .4', diffuse='.6 .6 .6', specular='.1 .1 .1')
    world = root.find('worldbody')
    ET.SubElement(world, 'light', pos='.2 -.5 1.5', dir='.2 .2 -1', diffuse='.8 .8 .8')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='2 2 .1', rgba='.24 .27 .30 1',
                  friction='.8 .01 .001')
    ET.SubElement(world, 'camera', name='overview', pos='1.05 -.85 .85',
                  xyaxes='.85 .52 0 -.28 .46 .84')
    ET.SubElement(world, 'camera', name='apparatus', pos='.91 -.48 .54',
                  xyaxes='.78 .62 0 -.30 .38 .875')
    pedestal_half = (SUPPORT_TOP - .022) / 2
    ET.SubElement(world, 'geom', name='pedestal', type='cylinder', size=numbers([.035, pedestal_half]),
                  pos=numbers([*STACK_XY, pedestal_half]), rgba='.18 .26 .32 1')
    ET.SubElement(world, 'geom', name='plate', type='cylinder', size='.075 .006',
                  pos=numbers([*STACK_XY, SUPPORT_TOP - .020]), rgba='.76 .80 .83 1', friction='.7 .01 .001')
    ET.SubElement(world, 'geom', name='stack_support', type='cylinder', size='.008 .007',
                  pos=numbers([*STACK_XY, SUPPORT_TOP - .007]), rgba='.40 .46 .50 1',
                  friction='.9 .01 .001', solref='.003 1', solimp='.95 .99 .001')

    wrist = root.find(".//body[@name='wrist_3_link']")
    attachment = wrist.find("site[@name='attachment_site']")
    tool = ET.SubElement(wrist, 'body', name='fork_clamp', pos=attachment.get('pos'),
                         quat=attachment.get('quat'), gravcomp='1')
    # Explicit inertials keep decorative labels from modifying tool inertia.
    ET.SubElement(tool, 'inertial', pos=numbers([TOOL_X / 2, 0, .06]), mass='.20', diaginertia='.001 .002 .002')
    common = dict(contype='1', conaffinity='1', friction='1.3 .02 .001',
                  solref='.003 1', solimp='.95 .99 .001', rgba='.55 .59 .64 1')
    ET.SubElement(tool, 'geom', name='clamp_connector', type='box',
                  pos=numbers([(TOOL_X - .10) / 2, 0, .018]),
                  size=numbers([(TOOL_X - .10) / 2 + .005, .022, .009]), **common)
    for sign in (-1, 1):
        ET.SubElement(tool, 'geom', name=f'clamp_guide_{sign}', type='cylinder',
                      pos=numbers([TOOL_X - .10, sign * .026, .07]), size='.004 .05', **common)
        ET.SubElement(tool, 'geom', name=f'lower_fork_{sign}', type='box',
                      pos=numbers([TOOL_X + (FORK_FORWARD - .10) / 2, sign * .014, TOOL_Z]),
                      size=numbers([(.10 + FORK_FORWARD) / 2, .003, .001]), **common)
    ET.SubElement(tool, 'geom', name='lower_fork_back', type='box',
                  pos=numbers([TOOL_X - .10, 0, TOOL_Z]), size='.006 .03 .003', **common)
    ET.SubElement(tool, 'site', name='fork_center', pos=numbers([TOOL_X, 0, TOOL_Z]),
                  size='.002', rgba='1 0 0 0', group='4')
    jaw = ET.SubElement(tool, 'body', name='upper_jaw', pos=numbers([0, 0, TOOL_Z - JAW_OPEN_GAP]))
    ET.SubElement(jaw, 'inertial', pos=numbers([TOOL_X - .04, 0, 0]), mass='.07',
                  diaginertia='.0001 .0002 .0002')
    ET.SubElement(jaw, 'joint', name='jaw_slide', type='slide', axis='0 0 1', limited='true',
                  range=f'0 {JAW_TRAVEL}', damping='3', armature='.002', solreflimit='.003 1')
    ET.SubElement(jaw, 'geom', name='upper_fork_back', type='box',
                  pos=numbers([TOOL_X - .10, 0, 0]), size='.006 .03 .003', **common)
    for sign in (-1, 1):
        ET.SubElement(jaw, 'geom', name=f'upper_fork_{sign}', type='box',
                      pos=numbers([TOOL_X + (FORK_FORWARD - .10) / 2, sign * .014, 0]),
                      size=numbers([(.10 + FORK_FORWARD) / 2, .003, .001]), **common)
    ET.SubElement(root.find('actuator'), 'position', name='jaw_motor', joint='jaw_slide',
                  kp='900', kv='20', ctrllimited='true', ctrlrange=f'0 {JAW_TRAVEL}',
                  forcelimited='true', forcerange='-18 18')

    for index, radius in enumerate(RADII):
        body = ET.SubElement(world, 'body', name=f'pancake_{index}',
                             pos=numbers([*STACK_XY, SUPPORT_TOP + BOSS_HALF + index * PITCH]))
        ET.SubElement(body, 'freejoint', name=f'pancake_free_{index}')
        rgba = [.88 - index * .035, .62 - index * .045, .29 - index * .035, 1]
        ET.SubElement(body, 'geom', name=f'pancake_disc_{index}', type='cylinder',
                      size=numbers([radius, DISC_HALF]), mass='.025', rgba=numbers(rgba),
                      friction='1.3 .02 .001', solref='.003 1', solimp='.95 .99 .001')
        ET.SubElement(body, 'geom', name=f'pancake_boss_{index}', type='cylinder',
                      size=numbers([.008, BOSS_HALF]), mass='.002', rgba='.46 .31 .16 1',
                      friction='1.3 .02 .001', solref='.003 1', solimp='.95 .99 .001')
        _digits(body, index + 1, radius)
    rendered_xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(rendered_xml, assets=assets)
    names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    arm = np.array([model.joint(name).id for name in names])
    tokens = np.array([model.joint(f'pancake_free_{i}').id for i in range(N)])
    jaw_id = model.joint('jaw_slide').id
    metadata = {
        'scene_version': 'copuzzle.pancake_contact_clamp.v1', 'robot': 'UR5e',
        'asset_commit': provenance['source_commit'], 'asset_license': provenance['license'],
        'n_tokens': N, 'radii_m': list(RADII), 'disc_thickness_m': 2 * DISC_HALF,
        'symmetric_boss_height_m': 2 * BOSS_HALF, 'stack_pitch_m': PITCH,
        'stack_center_xy': STACK_XY.tolist(), 'support_top_m': SUPPORT_TOP,
        'tool_offset_x_m': TOOL_X, 'tool_offset_z_m': TOOL_Z,
        'fork_forward_reach_m': FORK_FORWARD, 'withdraw_distance_m': WITHDRAW_DISTANCE,
        'jaw_travel_m': JAW_TRAVEL, 'jaw_force_limit_n': 18.,
        'grasp': 'physical parallel forks with frictional contacts; no token equality constraints',
        'simplifications': ['rigid sorting discs', 'symmetric central spacer bosses',
                            'arm gravity compensation', 'joint-position robot actuators'],
        'physics_dt': float(physics_dt),
    }
    digest = hashlib.sha256(rendered_xml.encode()).hexdigest()
    scene = PancakeScene(
        arm_joint_ids=arm, arm_qpos_ids=model.jnt_qposadr[arm], arm_dof_ids=model.jnt_dofadr[arm],
        arm_actuator_ids=np.array([model.actuator(name).id for name in
                                  ('shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3')]),
        tool_site_id=model.site('fork_center').id, tool_down_quat=np.array([0., 1., 0., 0.]),
        workspace_bounds=np.array([[.25, -.3, .13], [.76, .3, .60]]),
        home_qpos=np.array([-np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.]),
        home_tool_position=np.array([.45, -.11, .34]), jaw_joint_id=jaw_id,
        jaw_qpos_id=int(model.jnt_qposadr[jaw_id]), jaw_dof_id=int(model.jnt_dofadr[jaw_id]),
        jaw_actuator_id=model.actuator('jaw_motor').id,
        token_body_ids=np.array([model.body(f'pancake_{i}').id for i in range(N)]),
        token_qpos_ids=model.jnt_qposadr[tokens].copy(), token_dof_ids=model.jnt_dofadr[tokens].copy(),
        tool_geom_ids=np.array([i for i in range(model.ngeom) if model.geom_bodyid[i] in
                                (model.body('fork_clamp').id, model.body('upper_jaw').id)]),
        token_geom_ids=np.array([[model.geom(f'pancake_disc_{i}').id,
                                 model.geom(f'pancake_boss_{i}').id] for i in range(N)]),
        model_sha256=digest, metadata=metadata)
    return model, scene
