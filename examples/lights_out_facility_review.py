"""Render and replay a six-press Lights Out facility acceptance episode.

Run with MUJOCO_GL=egl on a headless machine. All frames come directly from
MuJoCo. Static rest/full-stroke views are explicitly geometric diagnostics;
the GIF is recorded from actual continuous robot controls and contact events.
This is an engineering review episode, not a training-data collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import mujoco
import numpy as np

from copuzzle.games.lights_out.physics import (
    ButtonSequencePolicy, load_physics_trajectory, record_episode, replay_trajectory,
)
from copuzzle.games.lights_out.physics.env import PhysicalPuzzleEnv
from copuzzle.games.lights_out.physics.scene import ASSET_DIRECTORY, BUTTON_TRAVEL
from copuzzle.games.lights_out.rules import grid_rule


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_review(env):
    model, data, scene = env.model, env.data, env.scene
    initial_qpos = data.qpos.copy()
    rows = []
    try:
        for depth in np.linspace(0., BUTTON_TRAVEL, 5):
            data.qpos[scene.button_qpos_ids] = depth
            mujoco.mj_forward(model, data)
            for index in range(env.rule.n_buttons):
                cap, stem = model.geom(f"button_cap_{index}").id, model.geom(f"button_stem_{index}").id
                base = model.geom(f"button_base_{index}").id
                sleeves = [g for g in range(model.ngeom)
                           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f"button_sleeve_{index}_")]
                distance = lambda a, b: float(mujoco.mj_geomDistance(model, data, int(a), int(b), 1., None))
                rows.append({
                    "button_index": index, "compression_m": float(depth),
                    "stem_base_clearance_m": distance(stem, base),
                    "minimum_stem_sleeve_clearance_m": min(distance(stem, g) for g in sleeves),
                    "minimum_cap_sleeve_clearance_m": min(distance(cap, g) for g in sleeves),
                    "maximum_sleeve_base_support_gap_m": max(abs(distance(g, base)) for g in sleeves),
                    "stem_z_range_m": (data.geom_xpos[stem, 2] + np.array([-1., 1.]) * model.geom_size[stem, 1]).tolist(),
                    "cap_z_range_m": (data.geom_xpos[cap, 2] + np.array([-1., 1.]) * model.geom_size[cap, 1]).tolist(),
                })
    finally:
        data.qpos[:] = initial_qpos
        mujoco.mj_forward(model, data)
    assert min(row["stem_base_clearance_m"] for row in rows) >= .0015
    assert min(row["minimum_stem_sleeve_clearance_m"] for row in rows) >= .0015
    assert min(row["minimum_cap_sleeve_clearance_m"] for row in rows) >= .0015
    assert max(row["maximum_sleeve_base_support_gap_m"] for row in rows) <= 1e-8
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rule = grid_rule(3, 3)
    sequence = [0, 8, 4, 2, 6, 1]
    state = np.random.default_rng(args.seed).integers(0, 2, rule.n_buttons, dtype=np.uint8)
    prefix_states = [state.copy()]
    for button in sequence:
        prefix_states.append(rule.transition(prefix_states[-1], button))
    goal = prefix_states[-1]
    if any(np.array_equal(goal, prefix) for prefix in prefix_states[:-1]):
        raise ValueError("A sequence prefix already reaches the final goal")
    with PhysicalPuzzleEnv(rule, max_episode_steps=1500, render_mode="rgb_array",
                           facility_profile="button_board_v2") as env:
        env.reset(seed=args.seed, options={"state": state, "goal": goal})
        configuration = env.configuration()
        write_json(args.output / "configuration.json", configuration)
        write_json(args.output / "geometry_measurements.json", {
            "purpose": "compiled geometry acceptance; static joint-position sweep, not a motion trajectory",
            "model_sha256": env.model_sha256, "measurements": geometry_review(env),
        })
        mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"), None)
        shutil.copyfile(ASSET_DIRECTORY / "LICENSE", args.output / "UR5e_LICENSE")
        shutil.copyfile(ASSET_DIRECTORY / "provenance.json", args.output / "UR5e_provenance.json")
        imageio.imwrite(args.output / "overview.png", env.render())
        closeup = mujoco.MjvCamera()
        closeup.lookat[:] = [.49, 0., .052]
        closeup.distance, closeup.azimuth, closeup.elevation = .50, 95, -16
        options = mujoco.MjvOption()
        options.sitegroup[:] = 0
        renderer = mujoco.Renderer(env.model, height=480, width=640)

        def frame(camera=closeup):
            renderer.update_scene(env.data, camera=camera, scene_option=options)
            return renderer.render().copy()

        try:
            imageio.imwrite(args.output / "closeup_rest.png", frame())
            env.data.qpos[env.scene.button_qpos_ids] = BUTTON_TRAVEL
            mujoco.mj_forward(env.model, env.data)
            imageio.imwrite(args.output / "closeup_fullstroke_static.png", frame())
            env.reset(seed=args.seed, options={"state": state, "goal": goal})
            overhead = mujoco.MjvCamera()
            overhead.lookat[:] = [.49, 0., .052]
            overhead.distance, overhead.azimuth, overhead.elevation = .48, 90, -70
            imageio.imwrite(args.output / "board_ids.png", frame(overhead))
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            counts = [int(np.sum((segmentation[:, :, 0] == cap)
                                 & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)))
                      for cap in env.scene.button_geom_ids]
            assert min(counts) >= 5
            assert not np.any(segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_SITE)
            write_json(args.output / "render_identity.json", {
                "source": "MuJoCo native object segmentation; IDs identify solid contact caps",
                "cap_geom_ids": env.scene.button_geom_ids.tolist(), "cap_pixel_counts": counts,
                "visual_button_labels": list(range(1, 10)), "action_indices": list(range(9)),
                "debug_site_pixels": int(np.sum(segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_SITE)),
            })
            executor = ButtonSequencePolicy(sequence, action_scale=env.action_scale)
            frames = []
            contact_saved = False

            def policy(observation):
                nonlocal contact_saved
                step = env.max_episode_steps - int(observation["steps_remaining"][0])
                if step % 2 == 0:
                    frames.append(env.render())
                if not contact_saved and observation["previous_event_counts"].any():
                    imageio.imwrite(args.output / "actual_probe_contact.png", frame())
                    contact_saved = True
                return executor(observation)

            episode = record_episode(env, policy, seed=args.seed,
                                     options={"state": state, "goal": goal}, metadata={
                "purpose": "facility geometry and six measured contact presses; engineering acceptance only",
                "requested_buttons": sequence, "controller": "observation-only ButtonSequencePolicy",
                "initial_state_distribution": "seeded iid binary board; explicit goal formed from full requested sequence",
                "no_earlier_prefix_reaches_goal": True,
                "fixture": "button_board_v2 solid caps, captured stems and hollow collision sleeves",
            })
            frames.append(env.render())
            imageio.imwrite(args.output / "actual_final.png", env.render())
            assert episode.terminated[-1] and episode.event_buttons.tolist() == sequence
            assert contact_saved
            trajectory_path = episode.save(args.output / "six_press_episode.npz")
            restored = load_physics_trajectory(trajectory_path)
            replay = replay_trajectory(env, restored)
            write_json(args.output / "replay.json", replay)
            imageio.mimsave(args.output / "six_press_sequence.gif", frames, duration=100, loop=0)
            report = {
                "purpose": "facility acceptance; not a benchmark or training dataset",
                "task_name": rule.name, "facility_profile": "button_board_v2",
                "model_sha256": env.model_sha256, "configuration_sha256": env.configuration_sha256,
                "seed": args.seed, "initial_board": state.tolist(), "goal": goal.tolist(),
                "requested_action_indices": sequence, "actual_action_indices": episode.event_buttons.tolist(),
                "displayed_button_labels": [button + 1 for button in sequence],
                "control_steps": episode.num_steps,
                "simulated_seconds": episode.num_steps * env.control_timestep,
                "success": bool(episode.terminated[-1]), "replay": replay,
                "gif_frame_count": len(frames), "gif_frame_interval_seconds": .10,
                "static_pose_images": ["closeup_rest.png", "closeup_fullstroke_static.png", "board_ids.png"],
                "actual_rollout_images": ["actual_probe_contact.png", "actual_final.png"],
                "compiled_model_format": "MuJoCo MJB; requires compatible MuJoCo version recorded in configuration",
                "artifacts": {path.name: sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()},
            }
            write_json(args.output / "manifest.json", report)
            print(json.dumps({key: report[key] for key in ("model_sha256", "success", "control_steps", "simulated_seconds", "actual_action_indices", "replay")}, indent=2))
        finally:
            renderer.close()


if __name__ == "__main__":
    main()
