"""Run the offline GCRL interface example after installing CoPuzzle.

    python examples/goal_conditioning.py

This checks data and environment interfaces; it does not train an RL agent.
Dataset generation and environment replay are benchmark-side demonstrations.
An offline learner should load fixed training data and only use the environment
for evaluation after training.
"""
from __future__ import annotations

import json

import numpy as np

from copuzzle import get_rule, make_env
from copuzzle.games.lights_out.presets import TASK_PRESETS
from copuzzle.games.lights_out.data import GoalSampler, generate_dataset, relabel_transitions


def demonstrate(task: str) -> dict:
    # Benchmark preparation: record operations without choosing a collection goal.
    dataset = generate_dataset(get_rule(task), num_episodes=8, horizon=30, seed=0)
    start, successor = dataset.states[0, :2]
    future_states = dataset.states[0, 1:]
    alternatives = future_states[
        np.any(future_states != start, axis=1)
        & np.any(future_states != successor, axis=1)
    ]
    if len(alternatives) == 0:
        raise ValueError("The example needs a second distinct future board")

    # Attach two goals to exactly the same (state, action, next_state).
    # Both goals are observed later in this trajectory and therefore reachable.
    goals = np.stack([successor, alternatives[0]])
    comparison = relabel_transitions(dataset, [0, 0], [0, 0], goals)
    np.testing.assert_array_equal(comparison["rewards"], [0.0, -1.0])
    np.testing.assert_array_equal(comparison["terminated"], [True, False])

    # Replay only to verify that environment and offline labels agree.
    env = make_env(task, max_episode_steps=30)
    try:
        for i, goal in enumerate(goals):
            observation, _ = env.reset(options={"state": start, "goal": goal})
            np.testing.assert_array_equal(observation["goal"], goal)
            assert set(observation) == {"board", "goal", "previous_event"}
            observation, reward, terminated, truncated, _ = env.step(
                int(comparison["actions"][i])
            )
            np.testing.assert_array_equal(observation["board"], successor)
            np.testing.assert_array_equal(observation["goal"], goal)
            np.testing.assert_array_equal(observation["previous_event"], comparison["events"][i])
            assert reward == comparison["rewards"][i]
            assert terminated == comparison["terminated"][i]
            assert truncated == comparison["truncated"][i]
    finally:
        env.close()

    # Learner side: reuse this sampler across updates. It queries no rule or env.
    sampler = GoalSampler(dataset, seed=7)
    batch = sampler.sample(32)
    policy_inputs = {
        "board": batch["observations"],
        "goal": batch["goals"],
        "previous_event": batch["previous_events"],
    }
    next_policy_inputs = {
        "board": batch["next_observations"],
        "goal": batch["goals"],
        "previous_event": batch["events"],
    }
    # These dicts use the same field meanings as env observations, with a batch
    # dimension. A separate algorithm project converts them to its tensors and
    # uses actions, rewards and bootstrap_masks in its chosen learning objective.
    np.testing.assert_array_equal(policy_inputs["goal"], next_policy_inputs["goal"])
    success = np.all(next_policy_inputs["board"] == policy_inputs["goal"], axis=1)
    np.testing.assert_array_equal(batch["terminated"], success)
    np.testing.assert_array_equal(batch["rewards"], np.where(success, 0.0, -1.0))
    np.testing.assert_array_equal(batch["bootstrap_masks"], (~success).astype(np.float32))
    return {
        "task": task,
        "policy_input_shapes": {key: list(value.shape) for key, value in policy_inputs.items()},
        "same_transition_rewards": comparison["rewards"].tolist(),
        "same_transition_terminated": comparison["terminated"].tolist(),
    }


if __name__ == "__main__":
    print("Offline GCRL interface example; no model is trained or scored.")
    for preset in TASK_PRESETS:
        print(json.dumps(demonstrate(preset.task_id), sort_keys=True))
