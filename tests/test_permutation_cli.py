"""Exercise the public environment and saved-data-to-evaluation workflow."""
import hashlib
import json

import gymnasium as gym
import numpy as np
import pytest

import copuzzle
from copuzzle.cli import main
from copuzzle.games.permutation.data import generate_dataset, load_dataset
from copuzzle.games.permutation.diagnostics import PositionMapLookup, evaluate_lookup
from copuzzle.games.permutation.evaluation import load_task_suite


@pytest.mark.parametrize("task,gym_id,n,actions", [
    ("pancake-4", "CoPuzzle/Pancake4-v0", 4, 3),
    ("pancake-6", "CoPuzzle/Pancake6-v0", 6, 5),
    ("pancake-8", "CoPuzzle/Pancake8-v0", 8, 7),
    ("topspin-8-4", "CoPuzzle/TopSpin8x4-v0", 8, 3),
])
def test_public_dispatch_registration(task, gym_id, n, actions):
    assert copuzzle.get_rule(task).n_actions == actions
    for env in (copuzzle.make_env(task), gym.make(gym_id)):
        try:
            observation, _ = env.reset(seed=5)
            assert env.observation_space.contains(observation)
            assert observation["board"].shape == (n,)
            assert observation["previous_event"].shape == (actions,)
            assert set(observation) == {"board", "goal", "previous_event", "action_mask"}
        finally:
            env.close()


def test_saved_collection_is_complete_disjoint_and_audited(tmp_path, capsys):
    output = tmp_path / "collection"
    main(["permutation", "collection", "--output", str(output), "--tasks", "pancake-4",
          "--episodes", "12", "--val-episodes", "4", "--per-depth", "2"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    for name, checksum in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum
    folder = output / "pancake-4"
    train = load_dataset(folder / "train.npz")
    train.validate(copuzzle.get_rule("pancake-4"))
    assert train.states.shape == (12, 31, 4)
    assert train.events.shape == (12, 30, 3)
    dev = load_task_suite(folder / "dev_tasks.json")
    test = load_task_suite(folder / "test_tasks.json")
    pairs = lambda suite: {(tuple(c["state"]), tuple(c["goal"])) for c in suite["cases"]}
    assert not pairs(dev) & pairs(test)
    for split in ("dev", "test"):
        report = json.loads((folder / f"{split}_oracle.json").read_text())
        assert report["summary"]["success_rate"] == 1
        assert report["summary"]["mean_excess_steps_on_success"] == 0
        assert "training_overlap" in report
    lookup = json.loads((folder / "lookup.json").read_text())
    assert lookup["single_step"]["exact_accuracy"] == 1
    assert all(s["endpoint_exact_accuracy"] == 1 for s in lookup["sequences"].values())
    before = (folder / "train.npz").read_bytes()
    with pytest.raises(SystemExit):
        main(["permutation", "collection", "--output", str(output)])
    assert (folder / "train.npz").read_bytes() == before
    assert capsys.readouterr().out


def test_cli_standalone_generation_validation_and_exclusion(tmp_path, capsys):
    folder = tmp_path / "data"
    main(["permutation", "generate", "--task", "topspin-5-3", "--output", str(folder),
          "--episodes", "6", "--val-episodes", "3", "--horizon", "10"])
    main(["permutation", "validate", str(folder / "train.npz"), "--check-rule"])
    for name in ("dev", "test"):
        args = ["permutation", "tasks", "--task", "topspin-5-3", "--depths", "1,2,3",
                "--per-depth", "2", "--output", str(folder / f"{name}.json")]
        if name == "test":
            args.extend(["--exclude-suite", str(folder / "dev.json")])
        main(args)
    main(["permutation", "evaluate", "--tasks", str(folder / "test.json"), "--policy", "oracle",
          "--training-data", str(folder / "train.npz"), "--output", str(folder / "oracle.json")])
    assert json.loads((folder / "oracle.json").read_text())["summary"]["success_rate"] == 1
    assert capsys.readouterr().out


def test_lookup_uses_ordered_position_maps_without_rule_queries(monkeypatch):
    rule = copuzzle.get_rule("pancake-4")
    train = generate_dataset(rule, 5, 20, 0, "train")
    test = generate_dataset(rule, 3, 20, 0, "val")
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline diagnostic must not query dynamics")
    monkeypatch.setattr(type(rule), "transition", forbidden)
    lookup = PositionMapLookup.fit(train)
    state = np.array([3, 1, 4, 2], dtype=np.int32)
    assert lookup.predict(lookup.predict(state, 0), 1).tolist() == [4, 3, 1, 2]
    assert lookup.predict(lookup.predict(state, 1), 0).tolist() == [1, 4, 3, 2]
    report = evaluate_lookup(train, test, num_examples=8)
    assert report["single_step"]["exact_accuracy"] == 1
    assert report["is_rl"] is False


def test_collection_rejects_infeasible_budget_before_writing(tmp_path):
    output = tmp_path / "bad"
    with pytest.raises(SystemExit):
        main(["permutation", "collection", "--output", str(output), "--horizon", "4"])
    assert not output.exists()
    with pytest.raises(SystemExit):
        main(["permutation", "collection", "--output", str(output), "--tasks", "pancake-4",
              "--per-depth", "37"])
    assert not output.exists()


def test_cli_diagnostic_supports_short_archives(tmp_path, capsys):
    main(["permutation", "generate", "--task", "pancake-2", "--output", str(tmp_path / "short"),
          "--episodes", "2", "--val-episodes", "2", "--horizon", "1"])
    main(["permutation", "diagnose", "--train", str(tmp_path / "short/train.npz"),
          "--test", str(tmp_path / "short/val.npz"), "--output", str(tmp_path / "lookup.json")])
    result = json.loads((tmp_path / "lookup.json").read_text())
    assert result["sequences"]["1"]["endpoint_exact_accuracy"] == 1
    assert result["training_data"]["content_sha256"] != result["validation_data"]["content_sha256"]
    assert capsys.readouterr().out
