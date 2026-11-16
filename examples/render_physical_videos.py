"""Re-render retained physical actions to SDR H.264 MP4, without a GIF roundtrip.

Example (headless): MUJOCO_GL=egl python examples/render_physical_videos.py
    --output artifacts/video-review-v1 --ffmpeg /path/to/ffmpeg

The original cameras, illumination, materials and simulation actions are retained.
RGB frames stream directly into FFmpeg. Original PNGs and decoded MP4 samples
provide a color comparison; every saved physical observation is checked on replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import mujoco
import numpy as np
from PIL import Image


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_case(task, root, trajectory=None):
    if task == 'lights-out':
        from copuzzle.games.lights_out.physics import make_physics_env, load_physics_trajectory
        path = trajectory or root / 'lightsout-board-v2/six_press_episode.npz'
        episode = load_physics_trajectory(path)
        c = episode.configuration
        env = make_physics_env(c['task_name'], max_episode_steps=c['max_episode_steps'],
                              physics_dt=c['physics_timestep'], control_dt=c['control_timestep'],
                              action_scale=c['action_scale'], press_depth=c['press_depth'],
                              release_depth=c['release_depth'], render_width=c['render_width'],
                              render_height=c['render_height'],
                              facility_profile=c['scene'].get('facility_profile', 'legacy_v1'))
        camera, camera_description = 'overview', {'name': 'overview', 'matches': 'six_press_sequence.gif'}
    elif task == 'topspin':
        from copuzzle.games.topspin.physics.env import TopSpinPhysicalEnv
        from copuzzle.games.topspin.physics.trace import load_trace
        path = trajectory or root / 'topspin-contact-v1/mixed.npz'
        episode = load_trace(path)
        c = episode.configuration
        env = TopSpinPhysicalEnv(max_episode_steps=c['max_episode_steps'],
                                physics_dt=c['physics_timestep'], control_dt=c['control_timestep'],
                                action_scale=c['action_scale'],
                                facility_profile=c.get('facility_profile', 'legacy_v1'))
        camera, camera_description = 'puzzle', {'name': 'puzzle'}
    else:
        from copuzzle.games.pancake.physics.env import PancakeManipulationEnv
        from copuzzle.games.pancake.physics.data import load_trajectory
        path = trajectory or root / 'pancake-contact-v1/episodes/1234-flip2.npz'
        episode = load_trajectory(path)
        c = episode.configuration
        env = PancakeManipulationEnv(max_episode_steps=c['max_episode_steps'],
                                    physics_dt=c['physics_timestep'], control_dt=c['control_timestep'],
                                    translation_scale=c['translation_scale'], rotation_scale=c['rotation_scale'],
                                    task_mode=c.get('task_mode', 'goal'),
                                    operation_prefix=c.get('operation_prefix'))
        description = {'lookat': [.43, 0, .33], 'distance': .85,
                       'azimuth_degrees': 135, 'elevation_degrees': -25}
        camera = mujoco.MjvCamera()
        camera.lookat[:] = description['lookat']
        camera.distance = description['distance']
        camera.azimuth = description['azimuth_degrees']
        camera.elevation = description['elevation_degrees']
        camera_description = description
    if env.configuration() != episode.configuration:
        env.close()
        raise ValueError(f'{task}: current model/configuration differs from retained trajectory')
    return path, episode, env, camera, camera_description


def compare_observation(actual, expected, index):
    maximum = 0.
    for key, values in expected.items():
        observed, reference = actual[key], values[index]
        if observed.shape != reference.shape or observed.dtype != reference.dtype:
            raise ValueError(f'{key}: shape/dtype differs during replay at {index}')
        if reference.dtype.kind == 'f':
            error = float(np.max(np.abs(observed - reference)))
            maximum = max(maximum, error)
            if not np.allclose(observed, reference, atol=1e-8, rtol=0):
                raise ValueError(f'{key}: replay error {error} at {index}')
        elif not np.array_equal(observed, reference):
            raise ValueError(f'{key}: replay differs at {index}')
    return maximum


def verify_video(ffmpeg, video, samples, width, height, expected_frames, output):
    # Decode the whole stream; the final progress record gives the decoded count.
    check = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostats',
                            '-i', str(video), '-map', '0:v:0', '-progress', 'pipe:1',
                            '-f', 'null', '-'], capture_output=True, text=True, check=True)
    counts = re.findall(r'^frame=(\d+)$', check.stdout, flags=re.MULTILINE)
    if not counts or int(counts[-1]) != expected_frames:
        raise ValueError('MP4 decoded frame count differs from rendered frames')
    probe = subprocess.run([ffmpeg, '-hide_banner', '-i', str(video), '-frames:v', '1',
                            '-vf', 'showinfo', '-f', 'null', '-'], capture_output=True, text=True, check=True)
    stream_lines = [line.strip() for line in probe.stderr.splitlines()
                    if 'Video:' in line or 'color_range:' in line or 'Duration:' in line]
    if not any('yuv420p(tv, bt709' in line for line in stream_lines):
        raise ValueError('Expected limited-range BT.709 YUV420 MP4 metadata')
    selected = sorted(samples)
    expression = '+'.join(f'eq(n,{index})' for index in selected)
    decoded = subprocess.run([
        ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', str(video),
        '-vf', f"select='{expression}',scale=in_color_matrix=bt709:in_range=tv:out_range=full,format=rgb24",
        '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1',
    ], capture_output=True, check=True)
    expected_bytes = len(selected) * width * height * 3
    if len(decoded.stdout) != expected_bytes:
        raise ValueError('Unexpected decoded RGB sample byte count')
    frames = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(len(selected), height, width, 3)
    comparisons = []
    for index, frame in zip(selected, frames):
        reference = samples[index]
        delta = frame.astype(np.float64) - reference.astype(np.float64)
        mse = float(np.mean(delta ** 2))
        name = f'{video.stem}-frame-{index:06d}-mp4.png'
        Image.fromarray(frame).save(output / name)
        comparisons.append({
            'frame_index': index,
            'original_png': f'{video.stem}-frame-{index:06d}-original.png',
            'decoded_mp4_png': name,
            'mean_absolute_rgb_error_0_to_255': float(np.mean(np.abs(delta))),
            'mean_signed_rgb_error': delta.mean(axis=(0, 1)).tolist(),
            'p99_absolute_channel_error': float(np.percentile(np.abs(delta), 99)),
            'psnr_db': None if mse == 0 else float(10 * np.log10(255 ** 2 / mse)),
        })
    return {'full_decode_passed': True, 'decoded_frames': expected_frames,
            'stream_metadata': stream_lines, 'sample_color_comparisons': comparisons}


def render_case(task, root, output, ffmpeg, width, height, crf, *, trajectory=None, label=None):
    source, episode, env, camera, camera_description = load_case(task, root, trajectory)
    label = label or task
    count = len(episode.actions)
    dt = float(episode.configuration['control_timestep'])
    fps = 1 / dt
    video = output / f'{label}.mp4'
    log = output / f'{label}-encode.log'
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-n',
               '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
               '-framerate', format(fps, '.12g'), '-i', 'pipe:0', '-an',
               '-vf', 'scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p',
               '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-profile:v', 'high',
               '-color_range', 'tv', '-colorspace', 'bt709', '-color_primaries', 'bt709',
               '-color_trc', 'bt709', '-movflags', '+faststart', str(video)]
    samples, maximum = {}, 0.
    sample_indices = {0, count // 2, count}
    renderer = process = None
    try:
        seed = episode.metadata['seed'] if task == 'pancake' else episode.metadata['reset_seed']
        observation, _ = env.reset(seed=seed,
                                   options=episode.metadata['reset_options'])
        maximum = compare_observation(observation, episode.observations, 0)
        # Only the framebuffer capacity changes. Simulation settings, material
        # colors, lights, exposure and the original camera are not modified.
        env.model.vis.global_.offwidth = max(width, env.model.vis.global_.offwidth)
        env.model.vis.global_.offheight = max(height, env.model.vis.global_.offheight)
        renderer = mujoco.Renderer(env.model, height=height, width=width)
        options = mujoco.MjvOption()
        options.sitegroup[:] = 0
        with log.open('xb') as error_stream:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=error_stream)
            for index in range(count + 1):
                if index:
                    observation, reward, terminated, truncated, info = env.step(episode.actions[index - 1].copy())
                    maximum = max(maximum, compare_observation(observation, episode.observations, index))
                    if (reward != episode.rewards[index - 1] or terminated != episode.terminated[index - 1]
                            or truncated != episode.truncated[index - 1]):
                        raise ValueError('Replay reward/end flags differ')
                    if task == 'lights-out':
                        rows = np.flatnonzero(episode.event_steps == index - 1)
                        events = [(int(episode.event_substeps[row]), int(episode.event_buttons[row])) for row in rows]
                        if events != [(item['substep'], item['button']) for item in info['event_log']]:
                            raise ValueError('Lights Out measured events differ')
                    elif task == 'topspin' and info['event_log'] != episode.infos[index]['event_log']:
                        raise ValueError('TopSpin measured events differ')
                renderer.update_scene(env.data, camera=camera, scene_option=options)
                frame = renderer.render().copy()
                if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
                    raise ValueError('Unexpected renderer RGB output')
                if index in sample_indices:
                    samples[index] = frame.copy()
                    Image.fromarray(frame).save(output / f'{label}-frame-{index:06d}-original.png')
                process.stdin.write(frame.tobytes())
                if index % 100 == 0:
                    print(f'{task}: rendered {index + 1}/{count + 1} frames', flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f'FFmpeg failed; see {log}')
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        if renderer is not None:
            renderer.close()
        env.close()
    verification = verify_video(ffmpeg, video, samples, width, height, count + 1, output)
    return {
        'task': task, 'video': video.name, 'video_sha256': sha256(video),
        'source_trajectory': str(source), 'source_sha256': sha256(source),
        'source_manifest_sha256': sha256(source.with_suffix('.json')),
        'configuration_sha256': episode.metadata['configuration_sha256'],
        'resolution': [width, height], 'fps': fps, 'frame_count': count + 1,
        'simulation_seconds': count * dt, 'video_seconds': (count + 1) / fps,
        'playback_speed': 1, 'camera': camera_description,
        'lighting_materials_exposure': 'unchanged from the source model',
        'pipeline': 'original MuJoCo RGB -> BT.709 limited-range YUV420 -> H.264 MP4; no GIF input',
        'max_replay_observation_error': maximum, 'encoder_command': command,
        'verification': verification,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts')
    parser.add_argument('--tasks', nargs='+', choices=['lights-out', 'topspin', 'pancake'],
                        default=['lights-out', 'topspin', 'pancake'])
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg'))
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=960)
    parser.add_argument('--crf', type=int, default=16)
    parser.add_argument('--trajectory', type=Path,
                        help='Replay a specified native trajectory; requires exactly one task')
    parser.add_argument('--label', help='Output filename stem; requires exactly one task')
    args = parser.parse_args()
    if not args.ffmpeg or not Path(args.ffmpeg).is_file():
        parser.error('Supply an installed FFmpeg executable with --ffmpeg')
    if min(args.width, args.height) < 2 or args.width % 2 or args.height % 2:
        parser.error('YUV420 requires positive even frame dimensions')
    if not 0 <= args.crf <= 51 or len(set(args.tasks)) != len(args.tasks):
        parser.error('CRF must be 0..51 and task names must be distinct')
    if (args.trajectory or args.label) and len(args.tasks) != 1:
        parser.error('--trajectory and --label require exactly one task')
    if args.label is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.label):
        parser.error('--label must contain only ASCII letters, digits, underscores or hyphens')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    version = subprocess.run([args.ffmpeg, '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    records = []
    for task in args.tasks:
        print(f'Rendering and verifying {task}...', flush=True)
        record = render_case(task, args.artifacts.resolve(), output, args.ffmpeg, args.width, args.height, args.crf,
                             trajectory=args.trajectory.resolve() if args.trajectory else None, label=args.label)
        records.append(record)
        with (output / f'{args.label or task}.json').open('x') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
        print(json.dumps({'task': task, 'seconds': record['video_seconds'], 'fps': record['fps'],
                          'replay_error': record['max_replay_observation_error']}), flush=True)
    manifest = {'schema_version': 'copuzzle.video_review.v1', 'status': 'complete',
                'purpose': 'original-speed native rendering and encoding color inspection',
                'ffmpeg': version, 'ffmpeg_sha256': sha256(args.ffmpeg),
                'script_sha256': sha256(__file__), 'mujoco_version': mujoco.__version__, 'videos': records,
                'files': {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    with (output / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    print(str(output / 'manifest.json'), flush=True)


if __name__ == '__main__':
    main()
