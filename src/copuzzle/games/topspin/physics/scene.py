"""A planar, closed-track 20/4 TopSpin tabletop with a passive turntable.

The geometry follows US4871173A: an endless channel filled with independently
movable numbered tokens, with four tokens in a straight rotating channel. All
motion is produced by contact. Token roll/pitch and vertical travel limits substitute for retaining
lips; tokens retain vertical dynamics and rest on colliding support surfaces; there are no welds, token actuators, or permutation-driven movements.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

TOKEN_RADIUS = .0117
TOKEN_HALF_HEIGHT = .008
TOKEN_Z = .060
ARC_RADIUS = .012 / np.sin(np.pi / 12)
HALF_STRAIGHT = .050
CENTER = np.array([.55, 0.])
TURN_CENTER = np.array([.55, ARC_RADIUS, .048])
HANDLE_RADIUS = .068
SOCKET_RADIUS = .005
PIN_RADIUS = .0032
SOCKET_BOTTOM = -.004
PROFILES = ('legacy_v1', 'engaged_v2')


def _n(values):
    return ' '.join(format(float(value), '.17g') for value in values)


def slots():
    """Four straight-window positions, six right, four lower, six left."""
    points = [[x, ARC_RADIUS] for x in np.array([-3, -1, 1, 3]) * .012]
    points += [[HALF_STRAIGHT + ARC_RADIUS * np.cos(theta), ARC_RADIUS * np.sin(theta)]
               for theta in np.deg2rad([75, 45, 15, -15, -45, -75])]
    points += [[x, -ARC_RADIUS] for x in np.array([3, 1, -1, -3]) * .012]
    points += [[-HALF_STRAIGHT + ARC_RADIUS * np.cos(theta), ARC_RADIUS * np.sin(theta)]
               for theta in np.deg2rad([-105, -135, -165, -195, -225, -255])]
    return np.column_stack((np.asarray(points) + CENTER, np.full(20, TOKEN_Z)))


def _robot():
    directory = Path(__file__).resolve().parents[2] / 'lights_out/physics/assets/ur5e'
    provenance = json.loads((directory / 'provenance.json').read_text())
    contents = {}
    for name, entry in provenance['files'].items():
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'UR5e asset integrity failure: {name}')
        contents[name] = data
    return ET.fromstring(contents['ur5e.xml']), {k: v for k, v in contents.items() if k.startswith('assets/')}, provenance


_SEGMENTS = {'0': 'abcedf', '1': 'bc', '2': 'abged', '3': 'abgcd', '4': 'fgbc',
             '5': 'afgcd', '6': 'afgecd', '7': 'abc', '8': 'abcdefg', '9': 'abfgcd'}


def _number(body, number, *, scale=1., offset_y=0.):
    text = str(number)
    width = .006
    for index, digit in enumerate(text):
        x = (index - (len(text) - 1) / 2) * .008
        bars = {'a': (x, .0046, width/2, .0006), 'g': (x, 0, width/2, .0006),
                'd': (x, -.0046, width/2, .0006),
                'f': (x-width/2, .0023, .0006, .0023), 'b': (x+width/2, .0023, .0006, .0023),
                'e': (x-width/2, -.0023, .0006, .0023), 'c': (x+width/2, -.0023, .0006, .0023)}
        for segment in _SEGMENTS[digit]:
            bx, by, sx, sy = bars[segment]
            ET.SubElement(body, 'geom', name=f'number_{number}_{index}_{segment}', type='box',
                          pos=_n([bx*scale, by*scale+offset_y, TOKEN_HALF_HEIGHT + .00025]), size=_n([sx*scale, sy*scale, .0002]),
                          rgba='.07 .09 .12 1', mass='0', contype='0', conaffinity='0')


def _rail(world, a, b, name, *, radius=.002, color='.18 .23 .29 1'):
    ET.SubElement(world, 'geom', name=name, type='capsule', fromto=_n([*a, .060, *b, .060]),
                  size=str(radius), rgba=color, friction='.12 .005 .001', solref='.006 1')
    delta=np.asarray(b)-np.asarray(a)
    ET.SubElement(world,'geom',name=name+'_support',type='box',
                  pos=_n([*(np.asarray(a)+np.asarray(b))/2,.055]),
                  size=_n([np.linalg.norm(delta)/2,radius,.003]),
                  euler=_n([0,0,np.arctan2(delta[1],delta[0])]),rgba=color,
                  friction='.12 .005 .001',solref='.006 1')


def build_scene(*, physics_dt=.002, facility_profile='engaged_v2'):
    if facility_profile not in PROFILES:raise ValueError('Unknown TopSpin facility profile')
    if isinstance(physics_dt, bool) or not np.isfinite(physics_dt) or not 0 < physics_dt <= .003:
        raise ValueError('physics_dt must lie in (0, .003]')
    root, assets, provenance = _robot()
    root.set('model', 'copuzzle_topspin20_planar_track')
    for body in root.iter('body'):
        body.set('gravcomp', '1')
    for key in root.findall('keyframe'):
        root.remove(key)
    root.find('option').attrib.update(timestep=str(physics_dt), integrator='implicitfast',
                                     cone='elliptic', impratio='10', iterations='80', tolerance='1e-10')
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='960')
    world = root.find('worldbody')
    ET.SubElement(world, 'light', pos='.4 -.5 1.3', dir='0 .3 -1', diffuse='.8 .8 .8')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='2 2 .1', rgba='.2 .23 .27 1')
    ET.SubElement(world, 'geom', name='topspin_housing', type='box', pos='.55 0 .010',
                  size='.132 .125 .010', rgba='.12 .26 .39 1')
    for side in (-1,1):
        ET.SubElement(world,'geom',name=f'housing_side_{side}',type='box',
                      pos=_n([CENTER[0]+side*.114,.019,.035]),size='.004 .089 .015',rgba='.12 .26 .39 1')
    for end,y in enumerate((-.068,.106)):
        ET.SubElement(world,'geom',name=f'housing_end_{end}',type='box',
                      pos=_n([CENTER[0],y,.035]),size='.110 .004 .015',rgba='.12 .26 .39 1')
    ET.SubElement(world,'geom',name='turntable_bearing_pedestal',type='cylinder',
                  pos=_n([*TURN_CENTER[:2],.032]),size='.012 .012',rgba='.3 .34 .38 1')
    # The stationary deck has a real circular opening, so its collision
    # surface never intersects the rotating disk or the disk's channel walls.
    for row, lower_y in enumerate(np.arange(-.066, .104, .002)):
        upper_y=lower_y+.002
        nearest_y=max(0.,abs((lower_y+upper_y)/2-ARC_RADIUS)-.001)
        cut=np.sqrt(max(0.,.051**2-nearest_y**2)) if nearest_y<.051 else 0.
        spans=[(-.115,-cut),(cut,.115)] if cut else [(-.115,.115)]
        for side,(left,right) in enumerate(spans):
            ET.SubElement(world,'geom',name=f'deck_{row}_{side}',type='box',
                          pos=_n([CENTER[0]+(left+right)/2,(lower_y+upper_y)/2,.051]),
                          size=_n([(right-left)/2,.001,.001]),rgba='.72 .79 .83 1',
                          friction='.1 .005 .001')
    # Low channel walls visibly surround and physically constrain all tokens.
    # The upper straight segment belongs to the rotating turntable.
    for side in (-1, 1):
        for radius in (ARC_RADIUS - .014, ARC_RADIUS + .014):
            angles = np.linspace(-np.pi/2, np.pi/2, 33)
            points = np.column_stack((side * (HALF_STRAIGHT + radius*np.cos(angles)), radius*np.sin(angles))) + CENTER
            for index, (a, b) in enumerate(zip(points[:-1], points[1:])):
                # Leave the turntable's circular sweep open at its two interfaces.
                if min(np.linalg.norm(a - TURN_CENTER[:2]), np.linalg.norm(b - TURN_CENTER[:2])) < .052:
                    continue
                _rail(world, a, b, f'arc_rail_{side}_{radius:.3f}_{index}')
        _rail(world, CENTER + [-HALF_STRAIGHT, -ARC_RADIUS + side*.014],
              CENTER + [HALF_STRAIGHT, -ARC_RADIUS + side*.014], f'lower_rail_{side}')
    # Solid board surrounding the rotary insert retains the ends of its
    # straight channel while it is turning; openings align with the loop.
    for index, angle in enumerate(np.linspace(0, 2*np.pi, 97)[:-1]):
        following=angle+2*np.pi/96
        if min(abs(np.sin(angle)),abs(np.sin(following))) < .30:
            continue
        a=TURN_CENTER[:2]+.052*np.array([np.cos(angle),np.sin(angle)])
        b=TURN_CENTER[:2]+.052*np.array([np.cos(following),np.sin(following)])
        _rail(world,a,b,f'turntable_surround_{index}',color='.16 .28 .37 1')
    turn = ET.SubElement(world, 'body', name='turntable', pos=_n(TURN_CENTER))
    ET.SubElement(turn, 'joint', name='turntable_yaw', type='hinge', axis='0 0 1',
                  limited='true', range=_n([0, np.pi]), damping='.04', frictionloss='.3', solreffriction='.004 1', solimpfriction='.9999 .99999 .00001',
                  solreflimit='.004 1', solimplimit='.999 .9999 .0001')
    ET.SubElement(turn, 'inertial', pos='0 0 0', mass='.12', diaginertia='.000075 .000075 .00015')
    ET.SubElement(turn, 'geom', name='turntable_plate', type='cylinder', size='.050 .004',
                  rgba='.57 .29 .69 1', mass='0', friction='.1 .005 .001')
    # The turntable is a slotted solid insert, not two disconnected rails:
    # its filled side banks physically close the fixed track when turned.
    for side in (-1,1):
        angle=np.arcsin(.0128/.0495)
        angles=np.linspace(angle,np.pi-angle,33)
        outline=np.column_stack((.0495*np.cos(angles),side*.0495*np.sin(angles)))
        vertices=np.vstack((np.column_stack((outline,np.full(len(outline),.004))),
                            np.column_stack((outline,np.full(len(outline),.026)))))
        mesh=f'turntable_bank_mesh_{side}'
        ET.SubElement(root.find('asset'),'mesh',name=mesh,vertex=_n(vertices.ravel()))
        ET.SubElement(turn,'geom',name=f'turntable_bank_{side}',type='mesh',mesh=mesh,
                      rgba='.44 .20 .56 1',mass='0',friction='.25 .005 .001',solref='.006 1')
    # A raised, visible crank reaches beyond the turntable; all rotation is
    # driven by the robot contacting this handle, with no turntable actuator.
    ET.SubElement(turn, 'geom', name='crank_stem', type='cylinder', pos='0 .028 .027',
                  size='.003 .024', rgba='.72 .75 .78 1', mass='0')
    ET.SubElement(turn, 'geom', name='crank_arm', type='box', pos='0 .048 .050',
                  size='.004 .024 .003', rgba='.72 .75 .78 1', mass='0')
    ET.SubElement(turn, 'geom', name='turntable_handle', type='cylinder', pos=_n([0, HANDLE_RADIUS, .042]),
                  size='.007 .012', rgba='.95 .58 .12 1', friction='.7 .005 .001', solref='.008 1')
    ET.SubElement(turn, 'site', name='handle_center', pos=_n([0, HANDLE_RADIUS, .042]), size='.001', rgba='0 0 0 0')
    if facility_profile=='engaged_v2':
        # Reuse one convex sector mesh for all tokens. A collection of 24
        # independent solid sectors leaves a genuinely hollow central socket;
        # a single convex hull would incorrectly fill the opening.
        angle=np.pi/24
        outline=np.array([[radius*np.cos(theta),radius*np.sin(theta)]
                          for radius,theta in [(SOCKET_RADIUS,-angle),(TOKEN_RADIUS,-angle),
                                               (TOKEN_RADIUS,angle),(SOCKET_RADIUS,angle)]])
        vertices=np.vstack([np.column_stack((outline,np.full(4,z)))
                            for z in (SOCKET_BOTTOM,TOKEN_HALF_HEIGHT)])
        ET.SubElement(root.find('asset'),'mesh',name='token_socket_sector',vertex=_n(vertices.ravel()))
    positions = slots()
    for index, point in enumerate(positions):
        token = ET.SubElement(world, 'body', name=f'token_{index+1}', pos=_n(point))
        ET.SubElement(token, 'joint', name=f'token_x_{index+1}', type='slide', axis='1 0 0', damping='.08')
        ET.SubElement(token, 'joint', name=f'token_y_{index+1}', type='slide', axis='0 1 0', damping='.08')
        ET.SubElement(token, 'joint', name=f'token_z_{index+1}', type='slide', axis='0 0 1',
                      limited='true', range='-.003 .003', damping='.03', solreflimit='.006 1')
        ET.SubElement(token, 'joint', name=f'token_yaw_{index+1}', type='hinge', axis='0 0 1', damping='.00003')
        if facility_profile=='legacy_v1':
            ET.SubElement(token, 'geom', name=f'token_geom_{index+1}', type='cylinder',
                          size=_n([TOKEN_RADIUS, TOKEN_HALF_HEIGHT]), mass='.012',
                          rgba='.95 .91 .76 1', friction='.3 .005 .001', solref='.006 1')
            _number(token, index+1)
        else:
            mass=.012
            radial=mass*(3*TOKEN_RADIUS**2+(2*TOKEN_HALF_HEIGHT)**2)/12
            axial=mass*TOKEN_RADIUS**2/2
            ET.SubElement(token,'inertial',pos='0 0 0',mass=str(mass),diaginertia=_n([radial,radial,axial]))
            ET.SubElement(token,'geom',name=f'token_geom_{index+1}',type='cylinder',
                          pos='0 0 -.006',size=_n([TOKEN_RADIUS,.002]),mass='0',
                          rgba='.85 .79 .61 1',friction='.3 .005 .001',solref='.006 1')
            for sector in range(24):
                ET.SubElement(token,'geom',name=f'token_socket_{index+1}_{sector}',type='mesh',
                              mesh='token_socket_sector',euler=_n([0,0,sector*2*np.pi/24]),mass='0',
                              rgba='.95 .91 .76 1',friction='.3 .005 .001',solref='.006 1',
                              solimp='.98 .999 .0001')
            _number(token,index+1,scale=.45,offset_y=.0074)
    wrist = root.find(".//body[@name='wrist_3_link']")
    attachment = wrist.find("site[@name='attachment_site']")
    tool = ET.SubElement(wrist, 'body', name='contact_tool', pos=attachment.get('pos'), quat=attachment.get('quat'), gravcomp='1')
    ET.SubElement(tool, 'geom', name='probe_shaft', type='cylinder', pos='0 0 .055', size='.009 .055',
                  mass='.07', rgba='.65 .69 .73 1', contype='1', conaffinity='1')
    if facility_profile=='legacy_v1':
        ET.SubElement(tool, 'geom', name='probe_tip', type='sphere', pos='0 0 .115', size='.010',
                      mass='.02', rgba='.18 .19 .21 1', friction='1 .01 .001', contype='1', conaffinity='1', solref='.008 1')
    else:
        ET.SubElement(tool,'geom',name='probe_tip',type='cylinder',pos='0 0 .117',
                      size=_n([PIN_RADIUS,.008]),mass='.02',rgba='.30 .34 .38 1',
                      friction='.6 .01 .001',contype='1',conaffinity='1',solref='.006 1',
                      solimp='.98 .999 .0001')
    ET.SubElement(tool, 'site', name='tool_tip', pos='0 0 .125', size='.001', rgba='0 0 0 0')
    ET.SubElement(world, 'camera', name='overview', pos='1.05 -.85 .8', xyaxes='.7 .7 0 -.35 .35 .87')
    ET.SubElement(world, 'camera', name='puzzle', pos='.55 -.20 .49', xyaxes='1 0 0 0 .89 .45')
    ET.SubElement(world, 'camera', name='top', pos='.55 0 .60', xyaxes='1 0 0 0 1 0')
    xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    arm_names = ('shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint')
    actuator_names = ('shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3')
    joints = np.array([model.joint(name).id for name in arm_names])
    token_joints = np.array([[model.joint(f'token_{axis}_{i}').id for axis in ('x','y','z','yaw')] for i in range(1,21)])
    scene = SimpleNamespace(
        arm_joint_ids=joints, arm_qpos_ids=model.jnt_qposadr[joints], arm_dof_ids=model.jnt_dofadr[joints],
        arm_actuator_ids=np.array([model.actuator(name).id for name in actuator_names]),
        tool_site_id=model.site('tool_tip').id,
        tool_geom_ids=np.array([model.geom('probe_shaft').id, model.geom('probe_tip').id]),
        tool_down_quat=np.array([0.,1.,0.,0.]), workspace_bounds=np.array([[.30,-.25,.049],[.75,.25,.45]]),
        home_qpos=np.array([-np.pi,-np.pi/2,np.pi/2,-np.pi/2,-np.pi/2,0.]),
        home_tool_position=np.array([.43,-.10,.30]), slots=positions,
        token_bodies=np.array([model.body(f'token_{i}').id for i in range(1,21)]),
        token_geoms=np.array([model.geom(f'token_geom_{i}').id for i in range(1,21)]),
        token_qpos=model.jnt_qposadr[token_joints], token_dofs=model.jnt_dofadr[token_joints],
        turntable_qpos=int(model.jnt_qposadr[model.joint('turntable_yaw').id]),
        turntable_dof=int(model.jnt_dofadr[model.joint('turntable_yaw').id]),
        handle_site=model.site('handle_center').id, handle_geom=model.geom('turntable_handle').id,
        model_sha256=hashlib.sha256(xml.encode()).hexdigest(), xml=xml,
        metadata={'scene_version':'copuzzle.topspin20_planar_track.v1', 'tokens':20, 'turntable_tokens':4,
                  'reference':'https://patents.google.com/patent/US4871173A/en',
                  'robot_asset_commit':provenance['source_commit'], 'robot_asset_license':provenance['license'],
                  'robot_asset_sha256':{name:entry['sha256'] for name,entry in provenance['files'].items()},
                  'simplifications':['token roll/pitch constrained and vertical travel limited to +/-3mm to model retaining groove lips; gravity and floor friction active',
                                     'passive revolute bearing with 0..180 degree mechanical stops',
                                     'gravity-compensated fixed-orientation UR5e probe'],
                  'motion_source':'direct robot contact; no puzzle actuators, welds or assigned token poses',
                  'bearing_frictionloss_nm':.3,'friction_impratio':10,
                  'parameter_status':'engineered simulation values, not measurements of a commercial toy',
                  'slot_positions':positions.tolist()})
    scene.facility_profile=facility_profile
    scene.handle_geom_ids=np.array([model.geom(name).id for name in
        (('turntable_handle',) if facility_profile=='legacy_v1' else ('turntable_handle','crank_arm','crank_stem'))])
    scene.geom_token_ids={int(model.geom(f'token_geom_{i}').id):i for i in range(1,21)}
    if facility_profile=='engaged_v2':
        scene.geom_token_ids.update({int(model.geom(f'token_socket_{i}_{sector}').id):i
                                    for i in range(1,21) for sector in range(24)})
        scene.metadata.update(scene_version='copuzzle.topspin20_planar_track.v2',
                              facility_profile=facility_profile,
                              contact_interface='visible cylindrical pin in individual blind sockets; collision-only engagement',
                              socket_radius=SOCKET_RADIUS, socket_bottom=SOCKET_BOTTOM,
                              pin_radius=PIN_RADIUS, socket_depth=TOKEN_HALF_HEIGHT-SOCKET_BOTTOM,
                              token_mass=.012, contact_friction=.6,
                              handle_contact_semantics='positive-load collision with the post, crank arm or crank stem')
    return model, scene


def track_coordinates(points):
    """Measured centerline arclength/distance for a clockwise racetrack.

    Order, rather than equal spacing, defines a real filled-track puzzle.
    """
    values=np.asarray(points,dtype=np.float64)[:,:2]-CENTER
    x,y=values[:,0],values[:,1]
    clipped=np.clip(x,-HALF_STRAIGHT,HALF_STRAIGHT)
    right_angle=np.clip(np.arctan2(y,x-HALF_STRAIGHT),-np.pi/2,np.pi/2)
    left_angle=np.arctan2(y,x+HALF_STRAIGHT)
    left_angle=np.where(left_angle>np.pi/2,left_angle-2*np.pi,left_angle)
    left_angle=np.clip(left_angle,-3*np.pi/2,-np.pi/2)
    projections=np.stack((np.column_stack((clipped,np.full(len(x),ARC_RADIUS))),
                          np.column_stack((HALF_STRAIGHT+ARC_RADIUS*np.cos(right_angle),ARC_RADIUS*np.sin(right_angle))),
                          np.column_stack((clipped,np.full(len(x),-ARC_RADIUS))),
                          np.column_stack((-HALF_STRAIGHT+ARC_RADIUS*np.cos(left_angle),ARC_RADIUS*np.sin(left_angle)))),axis=1)
    lengths=np.stack((clipped+HALF_STRAIGHT,
                      2*HALF_STRAIGHT+ARC_RADIUS*(np.pi/2-right_angle),
                      2*HALF_STRAIGHT+np.pi*ARC_RADIUS+HALF_STRAIGHT-clipped,
                      4*HALF_STRAIGHT+np.pi*ARC_RADIUS+ARC_RADIUS*(-np.pi/2-left_angle)),axis=1)
    distances=np.linalg.norm(values[:,None,:]-projections,axis=-1)
    best=np.argmin(distances,axis=1)
    return lengths[np.arange(len(x)),best],distances[np.arange(len(x)),best]
