"""Audit recorded TopSpin contact identity and relative translation offline.

The policy source must be a trusted local historical snapshot whose checksum
matches the trace. Exact action reconstruction establishes phase alignment.
No simulator, transition rule or goal solver is imported by this audit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def contact_segments(rows, dt):
    result = []
    for row in rows:
        ids = row["post_contacts"]
        if (result and ids == result[-1]["reported_token_ids"]
                and row["action_index"] == result[-1]["last_action_index"] + 1):
            result[-1]["last_action_index"] = row["action_index"]
            result[-1]["steps"] += 1
        else:
            result.append({"first_action_index": row["action_index"], "last_action_index": row["action_index"],
                           "steps": 1, "reported_token_ids": ids})
    for segment in result:
        segment["duration_seconds"] = segment["steps"] * dt
    return result


def relative_motion(observations, rows, selected, direction, dt):
    indices = np.array([row["action_index"] for row in rows])
    start, end = int(indices[0]), int(indices[-1]) + 1
    tangent = np.array([-direction, 0.])
    tool = observations["end_effector_position"][:, :2]
    token = observations["token_positions"][:, selected - 1, :2]
    relative = (tool - token) @ tangent
    increments = np.diff(relative)[indices]
    speeds = increments / dt
    touching = np.array([selected in row["post_contacts"] for row in rows])
    return {
        "selected_token_id": selected, "tangent_xy": tangent.tolist(),
        "tool_tangential_displacement_m": float((tool[end] - tool[start]) @ tangent),
        "selected_token_tangential_displacement_m": float((token[end] - token[start]) @ tangent),
        "relative_tangential_net_displacement_m": float(relative[end] - relative[start]),
        "relative_tangential_total_variation_m": float(np.abs(increments).sum()),
        "peak_relative_tangential_speed_m_s": float(np.abs(speeds).max()),
        "rms_relative_tangential_speed_m_s": float(np.sqrt(np.mean(speeds**2))),
        "variation_during_reported_selected_contact_m": float(np.abs(increments[touching]).sum()),
        "reported_contact_steps": int(touching.sum()), "reported_contact_fraction": float(touching.mean()),
        "post_step_offset_xy_norm_max_m": float(np.linalg.norm(tool[indices + 1] - token[indices + 1], axis=1).max()),
    }


def geometric_stadium_order(points, nominal_slots):
    """Independently decode the convex oval by polar order, not saved boards.

    Circle-fit the nominal right arc to obtain the track dimensions. Actual
    token centers are ordered clockwise about the oval center and anchored at
    the leftmost of the four tokens nearest the turntable center. This does
    not use the runtime piecewise-arclength decoder or a puzzle rule object.
    """
    points, slots = np.asarray(points)[:, :2], np.asarray(nominal_slots)[:, :2]
    center = slots.mean(axis=0)
    arc = slots[4:10]
    fit = np.linalg.lstsq(np.column_stack((2 * arc, np.ones(len(arc)))), (arc**2).sum(axis=1), rcond=None)[0]
    arc_center = fit[:2]
    radius = np.sqrt(fit[2] + np.dot(arc_center, arc_center))
    half_straight = abs(arc_center[0] - center[0])
    relative = points - center
    distance = np.abs(np.hypot(np.maximum(np.abs(relative[:, 0]) - half_straight, 0), relative[:, 1]) - radius)
    turn_center = slots[:4].mean(axis=0)
    window = np.argsort(np.linalg.norm(points - turn_center, axis=1))[:4]
    extent = float(np.linalg.norm(points[window] - turn_center, axis=1).max())
    if distance.max() > .0035 or extent > .040 or np.max(np.abs(points[window, 1] - turn_center[1])) > .0035:
        raise ValueError("Token centers do not form a docked four-token window on the physical oval")
    first = window[np.argmin(points[window, 0])]
    clockwise = np.argsort(-np.arctan2(relative[:, 1], relative[:, 0]))
    start = int(np.flatnonzero(clockwise == first)[0])
    return (np.roll(clockwise, -start) + 1).astype(int), {
        "maximum_centerline_distance_m": float(distance.max()), "window_radial_extent_m": extent,
        "method": "independent circle-fit geometry, clockwise polar ordering and physical window anchor",
    }


def audit_topspin(trace_path, policy_source_path):
    trace_path, policy_source_path = Path(trace_path).resolve(), Path(policy_source_path).resolve()
    metadata = json.loads(trace_path.with_suffix(".json").read_text())
    archive_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    if archive_sha != metadata["archive"]["sha256"]:
        raise ValueError("Archive checksum mismatch")
    policy_sha = hashlib.sha256(policy_source_path.read_bytes()).hexdigest()
    if policy_sha != metadata["metadata"]["sources"]["games/topspin/physics/scripted.py"]:
        raise ValueError("Policy snapshot checksum differs from the recorded source")
    spec = importlib.util.spec_from_file_location("reviewed_topspin_policy_snapshot", policy_source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    observations = {key[5:]: value for key, value in arrays.items() if key.startswith("obs__")}
    policy = module.SequencePolicy(metadata["metadata"]["user"]["intended_operations"],
                                   action_scale=metadata["configuration"]["action_scale"])
    dt = metadata["configuration"]["control_timestep"]
    rows, action_error = [], 0.
    for index, actual in enumerate(arrays["actions"]):
        inputs = {key: value[index].copy() for key, value in observations.items()}
        predicted = policy(inputs)
        action_error = max(action_error, float(np.max(np.abs(predicted - actual))))
        driver = policy.driver
        selected = getattr(driver, "token", None)
        rows.append({
            "action_index": index, "post_observation_index": index + 1, "operation_index": policy.index,
            "operation": policy.operations[policy.index] if policy.index < len(policy.operations) else "complete",
            "phase_after_command": getattr(driver, "phase", None),
            "selected_token": None if selected is None else selected + 1,
            "post_contacts": metadata["infos"][index + 1]["tool_token_contact_ids"],
        })
    if action_error != 0.:
        raise ValueError(f"Policy phase reconstruction does not match recorded actions: {action_error}")
    operations = []
    for operation_index, operation in enumerate(policy.operations):
        if not operation.startswith("shift"):
            continue
        group = [row for row in rows if row["operation_index"] == operation_index]
        push = [row for row in group if row["phase_after_command"] == "push"]
        if not push:
            raise ValueError("Expected a reconstructed push phase")
        selected = push[0]["selected_token"]
        if selected is None or any(row["selected_token"] != selected for row in push):
            raise ValueError("Policy did not retain one selected token during this push")
        contacts = sorted({token for row in push for token in row["post_contacts"]})
        phases = []
        for phase in dict.fromkeys(row["phase_after_command"] for row in group):
            phase_rows = [row for row in group if row["phase_after_command"] == phase]
            phases.append({"phase": phase, "first_action_index": phase_rows[0]["action_index"],
                           "last_action_index": phase_rows[-1]["action_index"]})
        operations.append({
            "operation_index": operation_index, "operation": operation,
            "first_action_index": group[0]["action_index"], "last_action_index": group[-1]["action_index"],
            "phase_intervals": phases,
            "push_first_action_index": push[0]["action_index"], "push_last_action_index": push[-1]["action_index"],
            "push_seconds": len(push) * dt, "selected_token_id": selected,
            "reported_push_contact_ids": contacts,
            "nonselected_contact_steps": sum(any(token != selected for token in row["post_contacts"]) for row in push),
            "exclusive_other_contact_steps": sum(bool(row["post_contacts"]) and selected not in row["post_contacts"] for row in push),
            "contact_segments_during_push": contact_segments(push, dt),
            "contact_segments_entire_operation": contact_segments(group, dt),
            "metrics": relative_motion(observations, push, selected, 1 if operation == "shift_right" else -1, dt),
        })
    return {
        "schema": "copuzzle.topspin_engagement_review.v1", "archive": str(trace_path),
        "archive_sha256": archive_sha, "policy_source_sha256": policy_sha,
        "action_reconstruction_max_absolute_error": action_error, "control_timestep_seconds": dt,
        "indexing": "zero-based action t has post observation/infos t+1",
        "measurement_limits": [
            "Contact IDs are control-boundary samples, not continuous physics-substep histories or force thresholds.",
            "Relative tangent motion uses tool-site and token-center translation; exact contact-surface slip cannot be recovered without contact points and rotational contributions.",
            "Policy phase is reconstructed from checksum-matching archived source and exact action agreement; it means phase after command preparation.",
        ],
        "operations": operations,
    }


def audit_engaged_topspin(trace_path):
    """Audit v2 controller intent against recorded physical observations.

    This reads the immutable archive and its explicit controller audit only.
    Force and contact-plane velocity measurements remain control-boundary
    samples; this function does not infer unobserved physics-substep behavior.
    """
    trace_path = Path(trace_path).resolve()
    header = json.loads(trace_path.with_suffix(".json").read_text())
    archive_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    if archive_sha != header["archive"]["sha256"]:
        raise ValueError("Archive checksum mismatch")
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    observations = {key[5:]: value for key, value in arrays.items() if key.startswith("obs__")}
    count = len(arrays["actions"])
    controller = header["metadata"]["user"]["controller_steps"]
    if len(controller) < count:
        raise ValueError("Controller audit omits executed actions")
    extra = len(controller) - count
    controller = controller[:count]
    intended = header["metadata"]["user"]["intended_operations"]
    dt = header["configuration"]["control_timestep"]
    nominal_clearance = (header["configuration"]["scene"]["socket_radius"]
                         - header["configuration"]["scene"]["pin_radius"])
    operations = []
    for operation_index, operation in enumerate(intended):
        if not operation.startswith("shift"):
            continue
        operation_rows = [(index, item) for index, item in enumerate(controller) if item["operation_index"] == operation_index]
        indices = [index for index, item in enumerate(controller)
                   if item["operation_index"] == operation_index and item["phase"] == "push"
                   and item["selected_token_id"] is not None]
        if not indices:
            continue
        selected_ids = {controller[index]["selected_token_id"] for index in indices}
        if len(selected_ids) != 1:
            raise ValueError("Controller changed selected identity within one shift")
        selected = selected_ids.pop()
        rows = [{"action_index": index, "post_contacts": header["infos"][index + 1]["tool_token_contact_ids"]}
                for index in indices]
        indices = np.asarray(indices, dtype=np.int64)
        masks = observations["tool_token_contact_mask"][indices + 1]
        for row, mask in zip(rows, masks):
            if row["post_contacts"] != (np.flatnonzero(mask) + 1).tolist():
                raise ValueError("Info contact identities disagree with observed masks")
        forces = observations["tool_token_contact_normal_force"][indices + 1, selected - 1]
        speeds = observations["tool_token_contact_tangent_speed"][indices + 1, selected - 1]
        offset = (observations["end_effector_position"][indices + 1, :2]
                  - observations["token_positions"][indices + 1, selected - 1, :2])
        loaded = forces > .05
        other = masks.copy()
        other[:, selected - 1] = 0
        metric = relative_motion(observations, rows, selected, 1 if operation == "shift_right" else -1, dt)
        operations.append({
            "operation_index": operation_index, "operation": operation, "selected_token_id": selected,
            "maximum_recorded_reacquisitions": max((int(item.get("reacquisitions", 0)) for _, item in operation_rows), default=0),
            "withdrawal_recovery_phases": [{"action_index": index, "phase": item["phase"]}
                                          for index, item in operation_rows if item["phase"] in ("reacquire_lift", "abort_lift", "failed")],
            "push_steps": len(indices), "push_first_action_index": int(indices[0]),
            "push_last_action_index": int(indices[-1]),
            "wrong_positive_load_identity_steps": int(np.any(other, axis=1).sum()),
            "selected_identity_contact_steps": int(masks[:, selected - 1].sum()),
            "selected_force_above_0_05N_steps": int(loaded.sum()),
            "unloaded_contact_steps": int((~masks[:, selected - 1].astype(bool)).sum()),
            "max_radial_center_offset_m": float(np.linalg.norm(offset, axis=1).max()),
            "nominal_radial_clearance_m": nominal_clearance,
            "max_radial_excess_over_nominal_clearance_m": float(np.linalg.norm(offset, axis=1).max() - nominal_clearance),
            "normal_force_quantiles_0_50_95_100N": np.quantile(forces, [0, .5, .95, 1]).tolist(),
            "loaded_contact_plane_speed_quantiles_50_95_100_m_s": np.quantile(speeds[loaded], [.5, .95, 1]).tolist() if loaded.any() else None,
            "sample_sum_max_loaded_contact_plane_speed_m": float(speeds[loaded].sum() * dt),
            "contact_segments": contact_segments(rows, dt), "center_relative_motion": metric,
        })
    events, record_errors = [], []
    previous_geometric_board, _ = geometric_stadium_order(observations["token_positions"][0], observations["slot_positions"][0])
    expected_operations = [{"reverse": "reverse_four"}.get(operation, operation) for operation in intended]
    for index, info in enumerate(header["infos"][1:]):
        for event in info["event_log"]:
            before, after = np.asarray(event["state_before"]), np.asarray(event["state_after"])
            operation = event["operation"]
            expected = (np.r_[before[:4][::-1], before[4:]] if operation == "reverse_four" else
                        np.roll(before, 1) if operation == "shift_right" else
                        np.roll(before, -1) if operation == "shift_left" else None)
            coherent = (np.array_equal(before, observations["board"][index])
                        and np.array_equal(after, observations["board"][index + 1]))
            transformation = expected is not None and np.array_equal(after, expected)
            measured_board, geometry = geometric_stadium_order(observations["token_positions"][index + 1],
                                                                observations["slot_positions"][index + 1])
            physical_identity = (np.array_equal(before, previous_geometric_board)
                                 and np.array_equal(after, measured_board))
            clear = (bool(observations["tool_clear"][index + 1, 0])
                     and observations["end_effector_position"][index + 1, 2] > .15
                     and not observations["tool_token_contact_mask"][index + 1].any()
                     and not info["tool_handle_contact"])
            settled = (np.max(np.abs(observations["token_velocities"][index + 1])) < .015
                       and abs(observations["turntable_velocity"][index + 1, 0]) < .05)
            row = {"action_index": index, "operation": operation, "atomic_cost": event.get("atomic_cost"),
                   "operation_index": event.get("operation_index"), "full_20_token_transformation_matches": transformation,
                   "board_arrays_match_event": coherent, "physical_token_positions_match_event": physical_identity,
                   "physical_order": measured_board.tolist(), "geometric_review": geometry,
                   "tool_withdrawn": bool(clear), "settled": bool(settled)}
            events.append(row)
            previous_geometric_board = measured_board
            if not (coherent and transformation and physical_identity and clear and settled and event.get("atomic_cost") == 1):
                record_errors.append(row)
    reference = observations["board"][0].copy()
    for operation in intended:
        if operation == "reverse":
            reference = np.r_[reference[:4][::-1], reference[4:]]
        else:
            reference = np.roll(reference, 1 if operation == "shift_right" else -1)
    final_clear = (bool(observations["tool_clear"][-1, 0])
                   and observations["end_effector_position"][-1, 2] > .15
                   and not observations["tool_token_contact_mask"][-1].any()
                   and not header["infos"][-1]["tool_handle_contact"])
    try:
        final_geometric_board, final_geometry = geometric_stadium_order(observations["token_positions"][-1],
                                                                        observations["slot_positions"][-1])
        final_geometry["valid"] = True
    except ValueError as error:
        # An honestly failed, safely withdrawn attempt need not have reached
        # an indexed puzzle state. Preserve that result rather than dropping it.
        final_geometric_board = None
        final_geometry = {"valid": False, "reason": str(error)}
    return {
        "schema": "copuzzle.topspin_engaged_archive_review.v1", "archive": str(trace_path),
        "archive_sha256": archive_sha, "model_sha256": header["configuration"]["model_sha256"],
        "source_hashes": header["metadata"]["sources"], "control_steps": count,
        "controller_rows_beyond_executed_actions": extra, "collector_stop": header["metadata"]["collector_stop"],
        "intended_operations": intended, "operation_reviews": operations, "event_reviews": events,
        "event_record_errors": record_errors,
        "actual_event_sequence_matches_intention": [event["operation"] for event in events] == expected_operations,
        "final_full_20_token_board_matches_composition": bool(np.array_equal(observations["board"][-1], reference)),
        "final_physical_order_matches_composition": bool(final_geometric_board is not None and np.array_equal(final_geometric_board, reference)),
        "final_physical_order": None if final_geometric_board is None else final_geometric_board.tolist(),
        "final_geometric_review": final_geometry,
        "final_tool_clear": bool(final_clear), "final_board_valid": bool(observations["board_valid"][-1, 0]),
        "final_atomic_operation_count": int(observations["atomic_operation_count"][-1, 0]),
        "classified_event_count": sum(event["atomic_cost"] == 1 for event in events),
        "measurement_limits": [
            "All contact/force/slip values are control-boundary observations, not proof about every physics substep.",
            "The slip-speed sum is a sampled sum of the maximum contact-plane speed, not a material point's exact traveled distance.",
            "The radial clearance is nominal; convex-sector socket faces and solver compliance slightly modify measured offset.",
        ],
    }


def reconstruct_engaged_sequence(trace_path, policy_source_path):
    """Recover per-driver reasons using the exact saved Sequence/Audited policy.

    Nondefault driver parameters need an explicitly matching factory; this
    helper rejects any action or audit mismatch rather than guessing them.
    """
    trace_path, policy_source_path = Path(trace_path).resolve(), Path(policy_source_path).resolve()
    header = json.loads(trace_path.with_suffix(".json").read_text())
    digest = hashlib.sha256(policy_source_path.read_bytes()).hexdigest()
    if digest != header["metadata"]["sources"]["games/topspin/physics/scripted.py"]:
        raise ValueError("Policy snapshot checksum differs from the recorded source")
    if hashlib.sha256(trace_path.read_bytes()).hexdigest() != header["archive"]["sha256"]:
        raise ValueError("Archive checksum mismatch")
    spec = importlib.util.spec_from_file_location("reviewed_engaged_policy_snapshot", policy_source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    user = header["metadata"]["user"]
    sequence = module.SequencePolicy(user["intended_operations"], action_scale=header["configuration"]["action_scale"])
    policy = module.AuditedPolicy(sequence, noise_amplitude=user["action_noise_amplitude"],
                                 seed=user["action_noise_seed"], lift_disturbance=user["injected_lift_disturbance"])
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    observations = {key[5:]: value for key, value in arrays.items() if key.startswith("obs__")}
    error, seen, recoveries = 0., set(), []
    for index, action in enumerate(arrays["actions"]):
        predicted = policy({key: value[index].copy() for key, value in observations.items()})
        error = max(error, float(np.abs(predicted - action).max()))
        for event in getattr(sequence.driver, "audit", []) if sequence.driver is not None else []:
            identity = (sequence.index, event["step"], event["reason"])
            if identity not in seen:
                seen.add(identity)
                recoveries.append({"global_action_index": index, "operation_index": sequence.index, **event})
    audit_matches = policy.controller_steps == user["controller_steps"][:len(arrays["actions"])]
    if error != 0. or not audit_matches:
        raise ValueError("Policy factory/source does not exactly reconstruct the executed trace")
    return {
        "schema": "copuzzle.topspin_policy_reconstruction.v1", "archive": str(trace_path),
        "archive_sha256": header["archive"]["sha256"], "policy_source_sha256": digest,
        "policy_snapshot": str(policy_source_path), "actions_checked": len(arrays["actions"]),
        "max_action_error": error, "controller_steps_match_exactly": audit_matches,
        "recovered_per_driver_audit": recoveries,
        "method": "Offline policy execution on recorded observations; no environment or solver queries",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--policy-source", type=Path)
    parser.add_argument("--engaged", action="store_true", help="Audit v2 recorded controller_steps and force observations")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.engaged and args.policy_source is None:
        parser.error("Historical audit requires --policy-source")
    result = audit_engaged_topspin(args.trace) if args.engaged else audit_topspin(args.trace, args.policy_source)
    if args.engaged and args.policy_source is not None:
        result["policy_reconstruction"] = reconstruct_engaged_sequence(args.trace, args.policy_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "operations": len(result.get("operations", result.get("operation_reviews", [])))}))
