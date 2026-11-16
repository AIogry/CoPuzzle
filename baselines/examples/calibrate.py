"""Bounded development timing; run CPU/CUDA separately before scheduling jobs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def gpu_memory(gpu):
    result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            check=True, text=True, capture_output=True)
    return int(result.stdout.strip())


def own_gpu_memory():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                            check=True, text=True, capture_output=True)
    values = [int(row.split(",")[1]) for row in result.stdout.splitlines() if row.split(",")[0].strip() == str(os.getpid())]
    return max(values, default=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--max-existing-gpu-mib", type=int, default=8192)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    before = None
    if args.backend == "cuda":
        before = gpu_memory(args.gpu)
        if before > args.max_existing_gpu_mib:
            parser.error(f"GPU {args.gpu} already uses {before} MiB; refusing to start this calibration")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = args.backend
    import jax
    from copuzzle_baselines.config import Config, METHODS
    from copuzzle_baselines.runtime import train, versions

    args.output.mkdir(parents=True)
    started = time.perf_counter()
    rows = []
    for method in METHODS:
        config = Config(method=method, hidden_dims=(128, 128), batch_size=256)
        run_path = args.output / method
        run = train(args.learner, run_path, config=config, seed=0, steps=512,
                    checkpoint_every=512, log_every=128)
        metrics = [json.loads(line) for line in (run_path / "metrics.jsonl").read_text().splitlines()]
        warm = next(item for item in metrics if item["step"] == 128)
        final = metrics[-1]
        row = {"method": method, "updates": 512, "train_seconds": run["train_seconds"],
               "warmup_through_step": 128,
               "steady_seconds_per_update": (final["seconds"] - warm["seconds"]) / 384,
               "final_loss": final["loss"],
               "observed_process_gpu_mib_after_fit": own_gpu_memory() if args.backend == "cuda" else None}
        rows.append(row)
        print(json.dumps(row), flush=True)
    host_rss = None
    if platform.system() == "Linux":
        import resource
        host_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {"purpose": "development hardware calibration; no formal fits", "backend": jax.default_backend(),
              "devices": [str(device) for device in jax.devices()], "versions": versions(),
              "gpu_index": args.gpu if args.backend == "cuda" else None, "gpu_mib_before": before,
              "preallocation": False if args.backend == "cuda" else None,
              "gpu_memory_measurement": "post-fit snapshots, not a sampled peak",
              "host_max_rss_kib": host_rss,
              "calibration_seconds_after_device_import": time.perf_counter() - started,
              "hidden_dims": [128, 128], "batch_size": 256, "results": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()
