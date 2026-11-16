"""Collect and replay one genuine contact flip; optionally render its actions.

Example: MUJOCO_GL=egl python -m copuzzle.games.pancake.physics.demo \
    --output artifacts/pancake-contact-demo --prefix 2 --render
"""
from pathlib import Path
import argparse
import json

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', type=int, choices=(1, 2, 3, 4), default=2,
                        help='Prefix 1 runs a physical execution diagnostic; it leaves the unsigned permutation unchanged.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args(argv)
    from .env import PancakeManipulationEnv
    from .scripted import PancakeFlipPolicy
    from .data import record_trajectory, load_trajectory, replay_trajectory
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f'prefix-{args.prefix}'
    mode = {'task_mode': 'operation', 'operation_prefix': 1} if args.prefix == 1 else {}
    env = PancakeManipulationEnv(render_mode='rgb_array' if args.render else None, **mode)
    try:
        state = np.arange(1, 5, dtype=np.int32)
        goal = state.copy()
        goal[:args.prefix] = goal[:args.prefix][::-1]
        episode = record_trajectory(env, PancakeFlipPolicy(args.prefix), seed=args.seed,
            options={'state': state, 'goal': goal},
            metadata={'purpose': 'contact-only scripted demonstration', 'prefix': args.prefix},
            on_policy_error='record')
        path = episode.save(stem.with_suffix('.npz'))
        report = replay_trajectory(env, load_trajectory(path))
        report['success'] = bool(episode.terminated[-1])
        report['task_mode'] = env.task_mode
        report['permutation_changed'] = not np.array_equal(episode.observations['board'][0], episode.observations['board'][-1])
        if args.prefix == 1:
            report['execution_success'] = episode.infos[-1]['execution_success']
            report['meaning'] = 'One-disc pick/lift/flip/replace execution; no permutation-solving improvement.'
        report['trajectory'] = str(path)
        if args.render:
            import imageio.v2 as imageio
            import mujoco
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [.43, 0., .33]
            camera.distance, camera.azimuth, camera.elevation = .85, 135, -25
            render = lambda: env.render(camera=camera)
            obs, _ = env.reset(seed=episode.metadata['seed'], options=episode.metadata['reset_options'])
            frames = [render()]
            imageio.imwrite(stem.with_suffix('.png'), frames[0])
            for i, action in enumerate(episode.actions):
                env.step(action)
                if (i + 1) % 20 == 0 or i == episode.num_steps - 1:
                    frames.append(render())
            duration_ms = 20 * env.control_dt * 1000 / 2
            imageio.mimsave(stem.with_suffix('.gif'), frames, duration=duration_ms, loop=0)
            report['preview'] = {'control_dt_seconds': env.control_dt, 'frame_stride': 20,
                                 'speedup': 2, 'frame_duration_ms': duration_ms, 'frames': len(frames)}
            imageio.imwrite(args.output / f'prefix-{args.prefix}-final.png', frames[-1])
        (args.output / f'prefix-{args.prefix}-replay.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
