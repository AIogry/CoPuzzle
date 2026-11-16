"""UR5e-driven indexed fixtures with real dynamic token bodies.

The fixture uses ideal rack/gear linkages and switchable docking welds. This is
a reduced mechanical model, not frictional grasping or a CAD model of a toy.
Tokens move through integration under carrier constraints; no token pose is
assigned during an operation. The existing licensed UR5e and probe are reused.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from copuzzle.games.lights_out.rules import grid_rule
from copuzzle.games.lights_out.physics.scene import build_scene as button_scene, _robot_assets

STROKE = .04
RACK_TRAVEL = .052
RING_RADIUS = .085


def _numbers(values):
    return " ".join(format(float(v), '.17g') for v in values)


def fixture_geometry(rule):
    if rule.name == 'topspin-8-4':
        center = np.array([.58, 0., .22])
        angles = 2 * np.pi * np.arange(8) / 8
        slots = center + RING_RADIUS * np.stack((np.cos(angles), np.sin(angles), np.zeros(8)), axis=1)
        axis = [np.cos(3 * np.pi / 8), np.sin(3 * np.pi / 8), 0.]
        return slots, [center] * 3, [[0, 0, 1], [0, 0, 1], axis], [-np.pi / 4, np.pi / 4, np.pi]
    if rule.name == 'pancake-4':
        slots = np.array([[.59, 0., .27 - .045 * i] for i in range(4)])
        return slots, [slots[:k].mean(axis=0) for k in (2, 3, 4)], [[0, 1, 0]] * 3, [np.pi] * 3
    raise ValueError('Physical fixtures currently support topspin-8-4 and pancake-4')


def build_scene(rule, *, physics_dt=.002):
    slots, centers, axes, angles = fixture_geometry(rule)
    # Reuse the exact same verified robot and contact probe, then replace the
    # button apparatus with rack inputs, geared carriers and captured tokens.
    # Preserve the archived experimental fixture; new button-board geometry
    # belongs to the separate Lights Out v2 facility.
    base_model, base_scene = button_scene(grid_rule(1, 3), physics_dt=physics_dt,
                                         facility_profile="legacy_v1")
    with tempfile.TemporaryDirectory(prefix='copuzzle-fixture-') as temporary:
        path = Path(temporary) / 'robot.xml'
        mujoco.mj_saveLastXML(str(path), base_model)
        root = ET.fromstring(path.read_text())
    root.set('model', rule.name + '_ur5e_indexed_fixture')
    world = root.find('worldbody')
    world.remove(world.find("geom[@name='button_panel']"))
    rest = np.array([[.38, -.12 + .12 * i, .10] for i in range(3)])
    for i in range(3):
        base = world.find(f"geom[@name='button_base_{i}']")
        base.set('pos', _numbers([*rest[i, :2], .025]))
        body = world.find(f"body[@name='button_{i}']")
        body.set('pos', _numbers(rest[i] - [0, 0, .006]))
        joint = body.find('joint')
        joint.attrib.update(range=f'0 {RACK_TRAVEL}', stiffness='70', springref='-.01', damping='4', armature='.002')
        body.find('geom').set('rgba', ('0.9 0.3 0.2 1', '0.2 0.7 0.9 1', '0.7 0.3 0.9 1')[i])
    equality = root.find('equality')
    if equality is None:
        equality = ET.SubElement(root, 'equality')
    # Carrier inertia is small but nonzero. Invisible linkages are represented
    # by joint equality constraints, whose reaction forces load the rack.
    for action, (center, axis, angle) in enumerate(zip(centers, axes, angles)):
        carrier = ET.SubElement(world, 'body', name=f'carrier_{action}', pos=_numbers(center), gravcomp='1')
        ET.SubElement(carrier, 'joint', name=f'carrier_hinge_{action}', type='hinge', axis=_numbers(axis),
                      limited='true', range=_numbers(sorted([0, angle])), damping='.002', armature='.00001')
        ET.SubElement(carrier, 'inertial', pos='0 0 0', mass='.005', diaginertia='.00001 .00001 .00001')
        ET.SubElement(carrier, 'geom', name=f'carrier_axis_{action}', type='capsule',
                      fromto=_numbers(np.concatenate((-np.asarray(axis)*.02, np.asarray(axis)*.02))),
                      size='.003', rgba='.6 .6 .6 .35', contype='0', conaffinity='0')
        ET.SubElement(equality, 'joint', name=f'gear_{action}', joint1=f'carrier_hinge_{action}',
                      joint2=f'button_slide_{action}', polycoef=_numbers([0, angle/STROKE, 0, 0, 0]),
                      active='false', solref='.004 1')
        ET.SubElement(equality, 'joint', name=f'carrier_lock_{action}', joint1=f'carrier_hinge_{action}',
                      polycoef='0 0 0 0 0', solref='.004 1')
        ET.SubElement(equality, 'joint', name=f'rack_lock_{action}', joint1=f'button_slide_{action}',
                      polycoef='0 0 0 0 0', active='false', solref='.004 1')
    colors = [(0.95,.30,.25), (.20,.75,.35), (.25,.45,.95), (.95,.75,.20),
              (.70,.35,.90), (.20,.80,.85), (.95,.50,.15), (.85,.55,.70)]
    for token, position in enumerate(slots):
        body = ET.SubElement(world, 'body', name=f'token_{token}', pos=_numbers(position))
        ET.SubElement(body, 'freejoint', name=f'token_free_{token}')
        ET.SubElement(body, 'geom', name=f'token_geom_{token}', type='cylinder', size='.013 .004',
                      mass='.008', rgba=_numbers([*colors[token], 1]), friction='.8 .01 .001',
                      solref='.004 1')
        ET.SubElement(body, 'site', name=f'token_site_{token}', size='.002')
        ET.SubElement(world, 'site', name=f'slot_{token}', pos=_numbers(position), size='.016',
                      rgba='.5 .5 .5 .15')
        ET.SubElement(equality, 'weld', name=f'dock_{token}', body1='world', body2=f'token_{token}',
                      solref='.004 1', solimp='.99 .999 .001')
        for action in range(3):
            ET.SubElement(equality, 'weld', name=f'carry_{action}_{token}', body1=f'carrier_{action}',
                          body2=f'token_{token}', active='false', solref='.004 1', solimp='.99 .999 .001')
    xml = ET.tostring(root, encoding='unicode')
    _, assets, provenance = _robot_assets()
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    # Base named indices are stable but explicitly resolve them after compilation.
    scene = replace(base_scene,
        tool_site_id=model.site('tool_tip').id,
        tool_geom_ids=np.array([model.geom('probe_shaft').id,model.geom('probe_tip').id]),
        button_rest_positions=rest,
        button_joint_ids=np.array([model.joint(f'button_slide_{i}').id for i in range(3)]),
        button_geom_ids=np.array([model.geom(f'button_cap_{i}').id for i in range(3)]),
        button_site_ids=np.array([model.site(f'button_top_{i}').id for i in range(3)]),
        home_tool_position=np.array([.38, 0., .32]),
        model_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        metadata={'scene_version':'copuzzle.indexed_permutation_fixture.v1', 'robot':'UR5e',
                  'asset_commit':provenance['source_commit'], 'asset_license':provenance['license'],
                  'simplifications':['ideal docking weld clamps', 'ideal rack/gear joint coupling',
                                     'gravity compensated carrier and arm', 'rigid tokens; no frictional grasp'],
                  'slot_positions':slots.tolist(), 'rack_stroke':STROKE,
                  'carrier_angles':angles, 'physics_dt':physics_dt})
    ids = lambda names: np.array([model.equality(name).id for name in names], dtype=np.int32)
    hinge_joints = np.array([model.joint(f'carrier_hinge_{i}').id for i in range(3)])
    fixture = {
        'slots':slots, 'angles':np.array(angles),
        'token_bodies':np.array([model.body(f'token_{i}').id for i in range(rule.n_tokens)]),
        'token_qpos':np.array([model.jnt_qposadr[model.joint(f'token_free_{i}').id] for i in range(rule.n_tokens)]),
        'token_dofs':np.array([model.jnt_dofadr[model.joint(f'token_free_{i}').id] for i in range(rule.n_tokens)]),
        'carrier_bodies':np.array([model.body(f'carrier_{i}').id for i in range(3)]),
        'hinge_joints':hinge_joints, 'hinge_qpos':model.jnt_qposadr[hinge_joints].copy(),
        'hinge_dofs':model.jnt_dofadr[hinge_joints].copy(),
        'gear':ids([f'gear_{i}' for i in range(3)]),
        'carrier_lock':ids([f'carrier_lock_{i}' for i in range(3)]),
        'rack_lock':ids([f'rack_lock_{i}' for i in range(3)]),
        'dock':ids([f'dock_{i}' for i in range(rule.n_tokens)]),
        'carry':np.array([ids([f'carry_{a}_{i}' for i in range(rule.n_tokens)]) for a in range(3)]),
    }
    return model, scene, fixture


def latch_current(model, data, equality_id):
    """Capture the current relative body pose without editing generalized state."""
    parent, child = model.eq_obj1id[equality_id], model.eq_obj2id[equality_id]
    rotation = data.xmat[parent].reshape(3, 3)
    relative = rotation.T @ (data.xpos[child] - data.xpos[parent])
    inverse, quaternion = np.empty(4), np.empty(4)
    mujoco.mju_negQuat(inverse, data.xquat[parent])
    mujoco.mju_mulQuat(quaternion, inverse, data.xquat[child])
    model.eq_data[equality_id, :3] = 0
    model.eq_data[equality_id, 3:6] = relative
    model.eq_data[equality_id, 6:10] = quaternion
    data.eq_active[equality_id] = True
