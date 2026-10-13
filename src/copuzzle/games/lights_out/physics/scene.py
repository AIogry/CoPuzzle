"""Self-contained UR5e scene with original spring buttons and a fixed probe.

The unchanged robot description and meshes come from a pinned MuJoCo Menagerie
revision; their BSD-3-Clause license and per-file hashes are packaged alongside
these assets. Scene geometry, button dynamics and physical layout are CoPuzzle
code. No OGBench, dm_control or local research checkout is used at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ..rules import PuzzleRule


_ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
_ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets" / "ur5e"
BUTTON_TRAVEL = 0.012
BUTTON_REST_HEIGHT = 0.080
BUTTON_SPACING = 0.065


def _readonly(value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _numbers(values):
    return " ".join(f"{float(value):.12g}" for value in values)


@dataclass(frozen=True)
class SceneDescription:
    """Named simulation indices and configuration, never a learned input schema.

    home_qpos has the six arm angles only. Each button has one slide joint;
    positive qpos means downward compression in metres. button_rest_positions
    are the top-surface positions at qpos=0, not the moving button centres.
    The probe site's local positive z points down at tool_down_quat.
    """
    arm_joint_ids: np.ndarray
    arm_qpos_ids: np.ndarray
    arm_dof_ids: np.ndarray
    arm_actuator_ids: np.ndarray
    button_joint_ids: np.ndarray
    button_qpos_ids: np.ndarray
    button_dof_ids: np.ndarray
    button_geom_ids: np.ndarray
    button_site_ids: np.ndarray
    tool_site_id: int
    tool_geom_ids: np.ndarray
    home_qpos: np.ndarray
    home_tool_position: np.ndarray
    tool_down_quat: np.ndarray
    workspace_bounds: np.ndarray
    button_rest_positions: np.ndarray
    model_sha256: str
    metadata: dict


@lru_cache(maxsize=1)
def _robot_assets():
    """Read and verify the pinned model once per process, including the license."""
    provenance = json.loads((ASSET_DIRECTORY / "provenance.json").read_text())
    files = {}
    for name, expected in provenance["files"].items():
        content = (ASSET_DIRECTORY / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected["sha256"]:
            raise ValueError(f"UR5e asset checksum mismatch: {name}")
        files[name] = content
    xml = files["ur5e.xml"].decode("utf-8")
    meshes = {name: content for name, content in files.items() if name.startswith("assets/")}
    return xml, meshes, provenance


def _layout(rule):
    if rule.grid_shape is not None:
        rows, cols = rule.grid_shape
    else:
        # Compact placement depends on N, never graph edges or display layout.
        cols = int(np.ceil(np.sqrt(rule.n_buttons)))
        rows = int(np.ceil(rule.n_buttons / cols))
    if rows > 6 or cols > 6:
        raise ValueError("The first physical scene supports layouts up to 6 rows and 6 columns")
    positions = np.empty((rule.n_buttons, 3), dtype=np.float64)
    for index in range(rule.n_buttons):
        row, col = divmod(index, cols)
        positions[index] = (
            0.49 + (row - (rows - 1) / 2) * BUTTON_SPACING,
            (col - (cols - 1) / 2) * BUTTON_SPACING,
            BUTTON_REST_HEIGHT,
        )
    return positions, (rows, cols)


def build_scene(rule: PuzzleRule, *, physics_dt=0.002):
    """Compile one fixed rule's physical layout and the packaged UR5e robot.

    Grid row/column order is preserved in a plane. Graph buttons use an
    independent rectangular placement with no drawn edges. The logical graph
    is supplied separately by the environment's public structure observation.
    """
    if not isinstance(rule, PuzzleRule):
        raise TypeError("rule must be a PuzzleRule")
    time_value = np.asarray(physics_dt)
    if (time_value.ndim != 0 or time_value.dtype.kind not in "iuf"
            or not np.isfinite(time_value) or not 0 < float(time_value) <= 0.005):
        raise ValueError("physics_dt must be finite and in (0, 0.005]")
    positions, layout_shape = _layout(rule)
    source_xml, assets, provenance = _robot_assets()
    root = ET.fromstring(source_xml)
    root.set("model", "copuzzle_ur5e_buttons")
    # A standard gravity-compensated arm holds still under zero Cartesian action.
    # Button gravity remains active and is resisted by each physical spring.
    for body in root.iter("body"):
        body.set("gravcomp", "1")
    root.find("option").attrib.update(
        timestep=str(float(physics_dt)), integrator="implicitfast", cone="elliptic",
        iterations="50", tolerance="1e-9", gravity="0 0 -9.81",
    )
    # The upstream home keyframe has six qpos entries, before buttons are added.
    for keyframe in root.findall("keyframe"):
        root.remove(keyframe)
    ET.SubElement(root, "statistic", center="0.3 0 0.25", extent="0.8")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", azimuth="135", elevation="-30", offwidth="640", offheight="480")
    ET.SubElement(visual, "headlight", diffuse="0.7 0.7 0.7", ambient="0.4 0.4 0.4", specular="0.2 0.2 0.2")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0.2 -0.5 1.8", dir="0.2 0.3 -1", diffuse="0.7 0.7 0.7")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.1",
                  rgba="0.22 0.25 0.29 1", friction="0.9 0.01 0.001")
    ET.SubElement(world, "camera", name="overview", pos="1.15 -1.15 0.95",
                  xyaxes="0.76 0.65 0 -0.32 0.37 0.87")
    board_size = (max(0.1, layout_shape[0] * BUTTON_SPACING / 2 + 0.02),
                  max(0.1, layout_shape[1] * BUTTON_SPACING / 2 + 0.02), 0.015)
    ET.SubElement(world, "geom", name="button_panel", type="box", pos="0.49 0 0.015",
                  size=_numbers(board_size), rgba="0.10 0.13 0.18 1")

    wrist = root.find(".//body[@name='wrist_3_link']")
    attachment = wrist.find("site[@name='attachment_site']")
    tool = ET.SubElement(wrist, "body", name="press_tool", pos=attachment.get("pos"),
                         quat=attachment.get("quat"), childclass="ur5e", gravcomp="1")
    # Explicit simple collision geometries avoid another licensed gripper asset.
    ET.SubElement(tool, "geom", name="probe_shaft", type="cylinder", size="0.009 0.055",
                  pos="0 0 0.055", mass="0.07", rgba="0.70 0.74 0.77 1",
                  contype="1", conaffinity="1", group="0")
    ET.SubElement(tool, "geom", name="probe_tip", type="sphere", size="0.010", pos="0 0 0.115",
                  mass="0.02", rgba="0.18 0.20 0.23 1", friction="0.7 0.01 0.001",
                  contype="1", conaffinity="1", group="0", solref="0.008 1")
    ET.SubElement(tool, "site", name="tool_tip", pos="0 0 0.125", size="0.002", rgba="1 0.5 0 0")

    for index, position in enumerate(positions):
        x, y, top_z = position
        ET.SubElement(world, "geom", name=f"button_base_{index}", type="cylinder",
                      size="0.025 0.010", pos=_numbers([x, y, 0.040]),
                      rgba="0.32 0.35 0.39 1")
        button = ET.SubElement(world, "body", name=f"button_{index}",
                               pos=_numbers([x, y, top_z - 0.006]))
        ET.SubElement(button, "joint", name=f"button_slide_{index}", type="slide",
                      axis="0 0 -1", limited="true", range=f"0 {BUTTON_TRAVEL}",
                      stiffness="200", springref="0", damping="2", armature="0.001",
                      solreflimit="0.004 1", solimplimit="0.99 0.999 0.0001")
        ET.SubElement(button, "geom", name=f"button_cap_{index}", type="cylinder",
                      size="0.021 0.006", mass="0.02", rgba="0.22 0.27 0.35 1",
                      contype="1", conaffinity="1", friction="0.8 0.01 0.001",
                      solref="0.008 1", solimp="0.95 0.99 0.001")
        ET.SubElement(button, "site", name=f"button_top_{index}", pos="0 0 0.006",
                      size="0.001", rgba="1 1 0 0")

    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    arm_joints = np.array([model.joint(name).id for name in _ARM_JOINTS])
    button_joints = np.array([model.joint(f"button_slide_{i}").id for i in range(rule.n_buttons)])
    bounds = np.array([[0.27, -0.25, 0.050], [0.71, 0.25, 0.42]])
    home = np.array([-np.pi, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
    metadata = {
        "scene_version": "copuzzle.ur5e_buttons.v1", "robot": "UR5e",
        "upstream_repository": provenance["source_repository"],
        "upstream_commit": provenance["source_commit"], "asset_license": provenance["license"],
        "asset_sha256": {name: row["sha256"] for name, row in provenance["files"].items()},
        "physics_dt": float(physics_dt), "button_travel": BUTTON_TRAVEL,
        "arm_gravity_compensation": True, "button_gravity_compensation": False,
        "button_stiffness": 200.0, "button_damping": 2.0, "button_mass": 0.02,
        "button_compression_sign": 1, "button_spacing": BUTTON_SPACING,
        "layout_shape": list(layout_shape),
        "layout_semantics": "grid rows along +x and columns along +y; graphs use an independent compact row-major placement",
        "button_rest_positions": positions.tolist(), "workspace_bounds": bounds.tolist(),
        "tool": "original fixed cylindrical probe with spherical contact tip",
        "tool_site": "bottom of probe sphere in the fixed downward orientation",
        "control": "Cartesian translation targets, fixed downward orientation, joint-position PD",
    }
    fingerprint = hashlib.sha256(xml.encode())
    fingerprint.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    scene = SceneDescription(
        arm_joint_ids=_readonly(arm_joints), arm_qpos_ids=_readonly(model.jnt_qposadr[arm_joints]),
        arm_dof_ids=_readonly(model.jnt_dofadr[arm_joints]),
        arm_actuator_ids=_readonly([model.actuator(name).id for name in _ARM_ACTUATORS]),
        button_joint_ids=_readonly(button_joints),
        button_qpos_ids=_readonly(model.jnt_qposadr[button_joints]),
        button_dof_ids=_readonly(model.jnt_dofadr[button_joints]),
        button_geom_ids=_readonly([model.geom(f"button_cap_{i}").id for i in range(rule.n_buttons)]),
        button_site_ids=_readonly([model.site(f"button_top_{i}").id for i in range(rule.n_buttons)]),
        tool_site_id=model.site("tool_tip").id,
        tool_geom_ids=_readonly([model.geom("probe_shaft").id, model.geom("probe_tip").id]),
        home_qpos=_readonly(home), home_tool_position=_readonly([0.49, 0.0, 0.25]),
        tool_down_quat=_readonly([0.0, 1.0, 0.0, 0.0]),
        workspace_bounds=_readonly(bounds), button_rest_positions=_readonly(positions),
        model_sha256=fingerprint.hexdigest(), metadata=metadata,
    )
    return model, scene
