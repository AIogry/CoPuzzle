"""Verify installed wheels outside the checkout without optional sibling/physics imports."""
import argparse
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forbid-source", type=Path, required=True)
    parser.add_argument("--with-baselines", action="store_true")
    args = parser.parse_args()
    class NoSiblingOrPhysics(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"copuzzle_lab", "mujoco", "dm_control"}:
                raise AssertionError(f"unexpected dependency: {fullname}")
    sys.meta_path.insert(0, NoSiblingOrPhysics())
    import copuzzle
    from copuzzle.benchmark.generation import write_bundle, validate_bundle
    assert not Path(copuzzle.__file__).resolve().is_relative_to(args.forbid_source.resolve())
    with tempfile.TemporaryDirectory(prefix="copuzzle-installed-") as temp:
        root = Path(temp)
        bundle = write_bundle(root / "data", "grid-3x3", "state_holdout", 0, max_per_cell=1)
        assert validate_bundle(bundle)["valid"]
        if args.with_baselines:
            import copuzzle_baselines
            from copuzzle_baselines.config import Config
            from copuzzle_baselines.runtime import train, evaluate
            assert not Path(copuzzle_baselines.__file__).resolve().is_relative_to(args.forbid_source.resolve())
            run = train(bundle / "learner", root / "fit", config=Config(method="dynamics", hidden_dims=(4,), batch_size=4), seed=0, steps=2)
            report = evaluate(root / "fit" / run["checkpoints"][-1]["file"], bundle, model_query_limit=8)
            assert report["learner_state_unchanged"]
    print(json.dumps({"installed_core": copuzzle.__file__, "learning_checked": args.with_baselines,
                      "sibling_and_physics_imports_blocked": True, "valid": True}))


if __name__ == "__main__":
    main()
