"""Replay checksum-bound native physical archives from an independently installed core."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualify(manifest, input_root, output, *, forbid_source=None):
    import copuzzle
    import mujoco
    import numpy
    from copuzzle.manipulation.runner import load_native, replay_native

    manifest, input_root, output = Path(manifest), Path(input_root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    package = Path(copuzzle.__file__).resolve().parent
    if forbid_source and package.is_relative_to(Path(forbid_source).resolve()):
        raise ValueError("physical qualification must use the independently installed wheel")
    inventory = json.loads(manifest.read_text())
    if inventory.get("schema") != "copuzzle.physical_history_inputs.v1":
        raise ValueError("unknown physical history manifest")
    cases = inventory["cases"]
    if len(cases) != 3 or {c["task"] for c in cases} != {"grid-3x3", "pancake-4", "topspin-20-4"}:
        raise ValueError("history qualification requires all three physical families exactly once")
    rows = []
    for case in cases:
        relative = Path(case["path"])
        path = (input_root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(input_root):
            raise ValueError("history input path escapes its root")
        started = time.perf_counter()
        row = dict(case)
        try:
            if sha(path) != case["sha256"]:
                raise ValueError("historical input checksum differs")
            metadata = (input_root / case["metadata_path"]).resolve()
            if metadata != path.with_suffix(".json") or not metadata.is_relative_to(input_root):
                raise ValueError("historical metadata path differs")
            if sha(metadata) != case["metadata_sha256"]:
                raise ValueError("historical metadata checksum differs")
            episode = load_native(case["task"], path)
            if (episode.metadata["configuration_sha256"] != case["configuration_sha256"]
                    or len(episode.actions) != case["control_steps"]):
                raise ValueError("historical task/configuration/length differs")
            row.update(status="passed", replay=replay_native(case["task"], episode))
        except Exception as error:
            row.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        row["elapsed_seconds"] = time.perf_counter()-started
        rows.append(row)
        print(json.dumps({"task": case["task"], "status": row["status"], "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
    result = {"schema": "copuzzle.physical_history_qualification.v1", "purpose": inventory["purpose"],
              "valid": all(r["status"] == "passed" for r in rows), "cases": rows,
              "manifest_sha256": sha(manifest), "script_sha256": sha(__file__), "installed_core": str(package),
              "versions": {"python": platform.python_version(), "numpy": numpy.__version__, "mujoco": mujoco.__version__},
              "sources": {p.relative_to(package).as_posix(): sha(p) for p in sorted(package.rglob("*.py"))},
              "scope": "saved-action/configuration reproduction on this runtime; no cross-platform or learned-control claim"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forbid-source", type=Path)
    args = parser.parse_args(argv)
    result = qualify(args.manifest, args.inputs, args.output, forbid_source=args.forbid_source)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
