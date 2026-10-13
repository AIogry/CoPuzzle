"""Record and replay a short real-contact diagnostic, optionally render a GIF.

Install copuzzle[physics], or copuzzle[render] for --gif. On Linux servers set
MUJOCO_GL=egl before launching Python. This is a scripted executor, not RL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import copuzzle
from copuzzle.games.lights_out.physics import (
    ButtonSequencePolicy, PhysicsGoalSampler, load_physics_trajectory,
    record_episode, replay_trajectory,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="grid-3x3")
    parser.add_argument("--buttons", default=None, help="Comma-separated IDs; default first,last,middle")
    parser.add_argument("--output", type=Path, required=True, help="New episode .npz path, with JSON sidecar")
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for path in [args.output, args.output.with_suffix(".json"), args.gif]:
        if path is not None and path.exists():
            parser.error(f"Output already exists: {path}")
    imageio = None
    if args.gif:
        import imageio.v2 as imageio
    env = copuzzle.make_physics_env(args.task, render_mode="rgb_array" if args.gif else None)
    try:
        count = env.rule.n_buttons
        sequence = ([0, count - 1, count // 2] if args.buttons is None else
                    [int(value) for value in args.buttons.split(",")])
        state = np.zeros(count, dtype=np.uint8)
        goal = state.copy()
        prefixes = [state.copy()]
        for button in sequence:
            goal = env.rule.transition(goal, button)
            prefixes.append(goal.copy())
        if not sequence or any(np.array_equal(prefix, goal) for prefix in prefixes[:-1]):
            parser.error("The initial board or an earlier sequence prefix already reaches this goal; choose a shorter sequence")
        executor = ButtonSequencePolicy(sequence, action_scale=env.action_scale)
        frames = []

        def policy(observation):
            # Capture one frame every two control intervals, independent of dynamics.
            if args.gif and int(observation["steps_remaining"][0]) % 2 == 0:
                frames.append(env.render())
            return executor(observation)

        episode = record_episode(
            env, policy, seed=args.seed, options={"state": state, "goal": goal},
            metadata={"controller": "scripted ButtonSequencePolicy", "requested_buttons": sequence,
                      "purpose": "physical contact diagnostic; not a training distribution"})
        if args.gif:
            frames.append(env.render())
        episode.save(args.output)
        restored = load_physics_trajectory(args.output)
        report = replay_trajectory(env, restored)
        report.update(task=args.task, success=bool(restored.terminated[-1]),
                      actual_buttons=restored.event_buttons.tolist(),
                      requested_buttons=sequence, trajectory=str(args.output))
        if report["success"]:
            batch = PhysicsGoalSampler(restored, seed=args.seed).sample(32)
            report["goal_batch_size"] = len(batch["actions"])
        if args.gif:
            args.gif.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(args.gif, frames, duration=100, loop=0)
            report["gif"] = str(args.gif)
        print(json.dumps(report, indent=2))
        if not report["success"] or report["actual_buttons"] != sequence:
            raise RuntimeError("The requested physical diagnostic did not complete exactly")
    finally:
        env.close()


if __name__ == "__main__":
    main()
