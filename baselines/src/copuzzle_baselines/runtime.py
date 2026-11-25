"""Training, immutable checkpoints and frozen evaluation with source identities."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import zipfile

import numpy as np

from .config import Config, canonical, fingerprint, integer
from .data import Sampler, Vocabulary, immutable, load_training


def sources():
    root = Path(__file__).parent
    names = ("__init__.py", "agents.py", "config.py", "data.py", "planning.py", "protocols.py", "runtime.py", "trajectories.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def versions():
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("numpy", "jax", "jaxlib", "flax", "optax")}}


def _json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def state_digest(agent):
    from flax import serialization
    return hashlib.sha256(serialization.msgpack_serialize(agent.state_dict())).hexdigest()


def save_checkpoint(path, agent, sampler, source_hashes, fit_context=None):
    from flax import serialization
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if sources() != source_hashes:
        raise RuntimeError("baseline source changed during training; checkpoint was not relabeled")
    state = serialization.msgpack_serialize(agent.state_dict())
    header = {"schema": "copuzzle.baseline_checkpoint.v2", "agent_state_sha256": hashlib.sha256(state).hexdigest(),
              "config": agent.config.to_dict(), "data": sampler.data.identity, "fit_seed": agent.seed,
              "step": agent.step, "sampler": sampler.state_dict(), "sources": source_hashes, "versions": versions(),
              "fit_context": fit_context}
    fd, name = tempfile.mkstemp(dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("header.json", canonical(header))
            archive.writestr("state.msgpack", state)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "step": agent.step}


def load_checkpoint(path):
    from flax import serialization
    from .agents import Agent
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != ["header.json", "state.msgpack"]:
            raise ValueError("unexpected checkpoint members")
        header = json.loads(archive.read("header.json"))
        state = archive.read("state.msgpack")
    keys = {"schema", "agent_state_sha256", "config", "data", "fit_seed", "step", "sampler", "sources", "versions", "fit_context"}
    if set(header) != keys or header["schema"] != "copuzzle.baseline_checkpoint.v2":
        raise ValueError("unsupported checkpoint header")
    if hashlib.sha256(state).hexdigest() != header["agent_state_sha256"]:
        raise ValueError("checkpoint state checksum mismatch")
    if header["sources"] != sources():
        raise ValueError("checkpoint source differs; use the run's saved source snapshot")
    config = Config.from_dict(header["config"])
    identity = header["data"]
    symbols = np.asarray(identity["symbols"], dtype=np.int32)
    if symbols.ndim != 1 or not len(symbols) or not np.array_equal(np.unique(symbols), symbols):
        raise ValueError("invalid fitted vocabulary")
    vocabulary = Vocabulary(immutable(symbols), integer(identity["state_width"], "state width", 1))
    agent = Agent(config, vocabulary.width, vocabulary.size, integer(identity["n_actions"], "n_actions", 1), header["fit_seed"])
    agent.restore(serialization.msgpack_restore(state))
    if agent.step != integer(header["step"], "step"):
        raise ValueError("checkpoint header step differs from saved learner")
    if canonical(header["sampler"]["config"]) != canonical(config.to_dict()):
        raise ValueError("sampler configuration differs from learner")
    if header["sampler"]["data_sha256"] != fingerprint(identity):
        raise ValueError("sampler training identity differs from checkpoint data")
    return agent, vocabulary, header


def train(learner_directory, output, *, config: Config, seed: int, steps: int,
          checkpoint_every=1000, log_every=100, resume=None, fit_context=None,
          trajectory_family=None):
    """No evaluator imports or file reads occur on the training path."""
    entered = time.perf_counter()
    from .agents import Agent
    seed, steps = integer(seed, "fit seed"), integer(steps, "steps", 1)
    checkpoint_every, log_every = integer(checkpoint_every, "checkpoint_every", 1), integer(log_every, "log_every", 1)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if trajectory_family is None:
        data = load_training(learner_directory, coverage=config.coverage, coverage_seed=config.coverage_seed)
        sampler = Sampler(data, config, seed + 10_000)
    else:
        if fit_context is not None:
            raise ValueError("development trajectories cannot enter the frozen formal study")
        from .trajectories import TrajectorySampler, load_trajectories
        data = load_trajectories(learner_directory, family=trajectory_family,
                                 coverage=config.coverage, coverage_seed=config.coverage_seed)
        sampler = TrajectorySampler(data, config, seed + 10_000)
    agent = Agent(config, data.width, data.vocabulary.size, data.n_actions, seed)
    parent = None
    if resume is not None:
        restored, _, header = load_checkpoint(resume)
        if restored.config != config or restored.seed != seed or canonical(header["data"]) != canonical(data.identity):
            raise ValueError("resume requires identical method, fit seed, training data and sampling configuration")
        if canonical(header["fit_context"]) != canonical(fit_context):
            raise ValueError("resume study context differs")
        sampler.restore(header["sampler"])
        agent = restored
        parent = {"file": str(Path(resume).resolve()), "sha256": hashlib.sha256(Path(resume).read_bytes()).hexdigest()}
    if steps <= agent.step:
        raise ValueError("steps must exceed the completed checkpoint step")
    source_hashes = sources()
    import jax
    if fit_context is not None:
        from .study.access import validate_fit_context
        validate_fit_context(fit_context, config.to_dict(), data.identity, source_hashes, versions(), seed, steps, jax.default_backend())
    output.mkdir(parents=True)
    snapshot = output / "source" / "copuzzle_baselines"
    snapshot.mkdir(parents=True)
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        target = snapshot / path.relative_to(Path(__file__).parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    if sources() != source_hashes:
        raise RuntimeError("source changed while snapshotting")
    run = {"schema": "copuzzle.baseline_run.v2", "purpose": "formal study fit" if fit_context else "development; not a frozen paper experiment",
           "status": "running", "config": config.to_dict(), "fit_seed": seed, "data": data.identity,
           "sources": source_hashes, "versions": versions(), "parent_checkpoint": parent,
           "start_step": agent.step, "completed_step": agent.step, "requested_steps": steps,
           "checkpoints": [], "train_seconds": 0.0,
           "schedule": {"checkpoint_every": checkpoint_every, "log_every": log_every},
           "execution": {"backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()]},
           "fit_context": fit_context, "setup_seconds": time.perf_counter() - entered}
    _json(output / "run.json", run)
    started = time.perf_counter()
    try:
        with (output / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            while agent.step < steps:
                metrics = agent.update(sampler.sample())
                if agent.step == run["start_step"] + 1 or agent.step % log_every == 0 or agent.step == steps:
                    record = {"step": agent.step, "seconds": time.perf_counter() - started, **metrics}
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                if agent.step % checkpoint_every == 0 or agent.step == steps:
                    run["checkpoints"].append(save_checkpoint(output / f"checkpoint-{agent.step:08d}.zip", agent, sampler, source_hashes, fit_context))
                    run["completed_step"] = agent.step
                    run["train_seconds"] = time.perf_counter() - started
                    _json(output / "run.json", run)
        run["status"] = "completed"
    except BaseException as error:
        run.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        run["completed_step"] = agent.step
        run["train_seconds"] = time.perf_counter() - started
        run["total_function_seconds"] = time.perf_counter() - entered
        run["sampling_counts"] = dict(sampler.counts)
        _json(output / "run.json", run)
    return run


def _check_data(learner_directory, header):
    path = Path(learner_directory)
    identity = header["data"]
    if identity["protocol"] == "copuzzle.symbolic-trajectories.dev1":
        raise ValueError("development trajectory checkpoints cannot use frozen R/C/G evaluation bundles")
    for name, key in (("train.npz", "train_sha256"), ("metadata.json", "metadata_sha256")):
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != identity[key]:
            raise ValueError("checkpoint and evaluation bundle have different training/action identities")


def _predict(agent, vocabulary, inputs, probe, batch_size):
    if agent.config.method != "dynamics":
        raise ValueError("policy-only baselines have no native R/C predictor")
    if probe == "R":
        boards, flags = [], []
        for start in range(0, len(inputs["board"]), batch_size):
            end = start + batch_size
            result, legal = agent.predict(vocabulary.encode(inputs["board"][start:end]), inputs["action"][start:end])
            boards.append(vocabulary.decode(result))
            flags.append(legal)
        return {"next_board": np.concatenate(boards), "legal": np.concatenate(flags)}
    if probe != "C":
        raise ValueError("prediction probe must be R or C")
    lengths, words = inputs["lengths"], inputs["actions"]
    result = np.full((*words.shape, vocabulary.width), -1, dtype=np.int32)
    current = vocabulary.encode(inputs["board"])
    for j in range(words.shape[1]):
        active = np.flatnonzero(lengths > j)
        for start in range(0, len(active), batch_size):
            rows = active[start:start + batch_size]
            # Never inject true intermediate states or override a model error.
            current[rows], _ = agent.predict(current[rows], words[rows, j])
            result[rows, j] = vocabulary.decode(current[rows])
    return {"prefix_boards": result}


def predict(checkpoint, learner_directory, probe, *, split="dev", batch_size=512, test_lock=None):
    from .protocols import public_api
    batch_size = integer(batch_size, "batch_size", 1)
    agent, vocabulary, header = load_checkpoint(checkpoint)
    _check_data(learner_directory, header)
    api = public_api(learner_directory)
    if api.metadata(learner_directory).get("purpose") == "benchmark" and split == "test":
        from .study.access import authorize_test
        authorize_test(test_lock, checkpoint, Path(learner_directory).parent, header)
    before = state_digest(agent)
    result = _predict(agent, vocabulary, api.load_probe(learner_directory, probe, split), probe, batch_size)
    if state_digest(agent) != before:
        raise RuntimeError("prediction changed learner state")
    return result


def evaluate(checkpoint, bundle, *, split="dev", model_query_limit=None, test_lock=None):
    """Evaluator-side entry point; training never imports this privileged code."""
    from .protocols import evaluator_api, public_api
    from .planning import ModelPlanner

    bundle = Path(bundle)
    generator, evaluator = evaluator_api(bundle)
    generator.validate_bundle(bundle)
    learner = bundle / "learner"
    api = public_api(learner)
    meta = api.metadata(learner)
    agent, vocabulary, header = load_checkpoint(checkpoint)
    _check_data(learner, header)
    authorization = None
    if meta.get("purpose") == "benchmark" and split == "test":
        from .study.access import authorize_test
        authorization = authorize_test(test_lock, checkpoint, bundle, header)
    before = state_digest(agent)
    learned_model = agent.config.method == "dynamics"
    limit = (32 if learned_model else 0) if model_query_limit is None else integer(model_query_limit, "model_query_limit")
    if not learned_model and limit != 0:
        raise ValueError("direct policies use the zero-explicit-model-query condition")
    if learned_model and limit == 0:
        raise ValueError("model-only planning has no zero-query actor; compare with the direct-policy controls")

    def forward(board, action):
        following, legal = agent.predict(vocabulary.encode(np.asarray(board)[None]), np.asarray([action], np.int32))
        return vocabulary.decode(following)[0], bool(legal[0])

    def batch_forward(boards, actions):
        following, legal = agent.predict(vocabulary.encode(boards), actions)
        return vocabulary.decode(following), legal

    def factory():
        if learned_model:
            return ModelPlanner(forward, agent.n_actions,
                                batch_predictor=batch_forward if meta["protocol"] == "copuzzle.rule-composition.v1" else None)
        return lambda observation, budget: int(agent.actions(vocabulary.encode(observation["board"][None]), vocabulary.encode(observation["goal"][None]))[0])

    start = time.perf_counter()
    report = {"schema": "copuzzle.baseline_evaluation.v2", "purpose": meta.get("purpose", "development"), "split": split,
              "method": agent.config.method, "fit_seed": agent.seed, "step": agent.step,
              "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              "bundle_manifest_sha256": hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
              "config": agent.config.to_dict(), "data": header["data"], "versions": versions(),
              "test_authorization": authorization, "fit_context": header["fit_context"],
              "R": {"supported": False}, "C": {"supported": False}}
    if learned_model:
        for probe in ("R", "C"):
            predictions = _predict(agent, vocabulary, api.load_probe(learner, probe, split), probe, 512)
            report[probe] = {"supported": True, **evaluator.score_predictions(bundle, probe, predictions, split=split)}
        if meta["protocol"] == "copuzzle.rule-composition.v1":
            from .study.diagnostics import composition_diagnostics
            report["C_context_diagnostic"] = composition_diagnostics(bundle, batch_forward, split=split,
                coverage=agent.config.coverage, coverage_seed=agent.config.coverage_seed)
    report["G"] = evaluator.evaluate_policy(bundle, factory, split=split, model_query_limit=limit)
    report["evaluation_seconds"] = time.perf_counter() - start
    report["learner_state_unchanged"] = state_digest(agent) == before
    if not report["learner_state_unchanged"]:
        raise RuntimeError("evaluation changed learner parameters/optimizer/step")
    return report
