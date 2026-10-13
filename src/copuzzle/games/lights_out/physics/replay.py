"""Replay stored continuous actions to audit a physical episode."""
from __future__ import annotations

import numpy as np


def replay_trajectory(env, episode, *, atol=1e-9):
    """Require matching configuration and compare every observation and event.

    The caller owns the environment. This is a simulator reproduction check,
    not a learned-policy evaluation. Matching MuJoCo version/configuration is
    required; cross-platform floating-point reproducibility is not promised.
    """
    episode.validate()
    if env.configuration_sha256 != episode.metadata["configuration_sha256"]:
        raise ValueError("Replay requires the recorded environment configuration")
    if isinstance(atol, (bool, np.bool_)) or not np.isscalar(atol) or not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    observation, _ = env.reset(seed=episode.metadata["reset_seed"],
                               options=episode.metadata["reset_options"])
    largest_error = 0.0

    def compare(actual, index):
        nonlocal largest_error
        expected = {name: values[index] for name, values in episode.observations.items()}
        expected.update(episode.structure.observation_fields())
        expected["button_rest_positions"] = episode.button_rest_positions
        if set(actual) != set(expected):
            raise ValueError("Replay observation fields differ")
        for name, value in expected.items():
            candidate = actual[name]
            if candidate.shape != value.shape or candidate.dtype != value.dtype:
                raise ValueError(f"Replay shape/dtype mismatch at observation {index}: {name}")
            if value.dtype.kind == "f":
                error = float(np.max(np.abs(candidate - value)))
                largest_error = max(largest_error, error)
                matches = np.allclose(candidate, value, atol=atol, rtol=0)
            else:
                matches = np.array_equal(candidate, value)
            if not matches:
                raise ValueError(f"Replay mismatch at observation {index}: {name}")

    compare(observation, 0)
    for step, action in enumerate(episode.actions):
        observation, reward, terminated, truncated, info = env.step(action)
        compare(observation, step + 1)
        mask = episode.event_steps == step
        events = [{"substep": int(substep), "button": int(button)}
                  for substep, button in zip(episode.event_substeps[mask], episode.event_buttons[mask])]
        if (info["event_log"] != events or reward != episode.rewards[step]
                or terminated != episode.terminated[step] or truncated != episode.truncated[step]):
            raise ValueError(f"Replay event/reward/end mismatch at control step {step}")
    return {"control_steps": episode.num_steps, "press_events": len(episode.event_buttons),
            "max_absolute_observation_error": largest_error, "atol": float(atol),
            "configuration_sha256": env.configuration_sha256}
