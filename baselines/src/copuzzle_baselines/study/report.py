"""Readable tables and optional scientific figures from a verified analysis."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import shutil


LEARNED = (("gcbc", 0, "GCBC"), ("gciql", 0, "GCIQL"),
           ("dynamics", 32, "Dynamics + planning (32)"), ("dynamics", 128, "Dynamics + planning (128)"))
REGIMES = ("transition_holdout", "state_holdout")
REGIME_NAMES = {"transition_holdout": "Transition holdout", "state_holdout": "State holdout"}
TASK_NAMES = {"grid-3x3": "Lights Out 3×3", "grid-4x4": "Lights Out 4×4", "pancake-6": "Pancake 6",
              "topspin-8-4": "TopSpin 8/4", "sliding-2x3": "Sliding 2×3", "hanoi-5": "Hanoi 5"}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _inside(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("analysis paths must stay inside the artifact")
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("analysis path escapes artifact")
    return result


def load_analysis(directory):
    directory = Path(directory)
    manifest = _read(directory / "manifest.json")
    if manifest.get("schema") != "copuzzle.study_analysis.v1":
        raise ValueError("unsupported analysis")
    if not {"rows.json", "statistics.json", "costs.json"}.issubset(manifest["files"]):
        raise ValueError("analysis manifest omits report input hashes")
    if manifest.get("synthetic") and manifest["role"] == "formal":
        raise ValueError("synthetic rendering data cannot be formal evidence")
    for name, digest in manifest["files"].items():
        if _hash(_inside(directory, name)) != digest:
            raise ValueError("analysis input checksum differs")
    if manifest["role"] == "formal" and manifest["split"] == "test" and not manifest.get("per_case_evidence_complete"):
        raise ValueError("formal results require complete per-case evidence")
    statistics, rows, costs = (_read(directory / name) for name in ("statistics.json", "rows.json", "costs.json"))
    if len(rows) != manifest["evaluation_jobs"] or dict(Counter(r["status"] for r in rows)) != manifest["outcomes"]:
        raise ValueError("analysis outcome inventory differs")
    return manifest, statistics, rows, costs


def _index(statistics):
    result = {}
    for row in statistics["estimates"]:
        key = row["regime"], row["metric"], row["scope"], row["series"]
        if key in result:
            raise ValueError("duplicate report estimate")
        result[key] = row
    return result


def _series(method, budget, coverage=1.0):
    return f"{method}/c{int(100*coverage):03d}/b{budget if budget is not None else 'exempt'}"


def _cell(row):
    if row is None:
        return "—"
    return f"{row['mean']:.3f} [{row['interval'][0]:.3f}, {row['interval'][1]:.3f}]"


def markdown(manifest, statistics, rows, costs, *, figures=False, discussion=False):
    final = manifest["role"] == "formal" and manifest["split"] == "test"
    title = "CoPuzzle v1 formal benchmark results" if final else "CoPuzzle analysis qualification — not final benchmark results"
    if manifest.get("synthetic"):
        title = "Synthetic rendering fixture — not experiment results"
    index = _index(statistics)
    lines = [f"# {title}", "", f"Study: `{manifest['study_sha256']}`. Split: `{manifest['split']}`.", "",
             f"Registered evaluation outcomes: {manifest['evaluation_jobs']}; completed: {manifest['outcomes'].get('completed', 0)}; "
             f"unavailable: {manifest['outcomes'].get('failed', 0)}. Unavailable runs remain in the denominators.", "",
             "Values are means with descriptive 95% paired hierarchical bootstrap intervals. "
             "The resampling unit is the data seed with nested fit seeds; cases and families are fixed. "
             f"Outer data seeds: {len(statistics['resampling']['data_seeds'])}; fit-seed slots: {len(statistics['resampling']['fit_seeds'])}. "
             "A small seed count provides limited uncertainty evidence. A dash marks an unsupported or out-of-scope estimate.", "",
             "## Full-coverage learned goal control", "",
             "G is equal-tier success. First average the two Lights Out instances, then weight the five families equally. "
             "No R/C/G composite score is used.", "",
             "| Method | Transition holdout | State holdout |", "|---|---:|---:|"]
    for method, budget, name in LEARNED:
        cells = [_cell(index.get((regime, "G", "core_family_macro", _series(method, budget)))) for regime in REGIMES]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    if figures:
        lines.extend(["", "![Full-coverage G](figures/G_primary.png)"])
    lines.extend(["", "## Goal-control references at full coverage", "",
                  "These references have distinct information/prior assumptions. The optimal oracle is budget-exempt; "
                  "the bounded true model uses the same planner and query limit as the learned dynamics.", "",
                  "| Reference | Transition holdout | State holdout |", "|---|---:|---:|"])
    for method, budget, name in (("random", 0, "Random actions"), ("lookup", 32, "Empirical lookup / 32"),
                               ("lookup", 128, "Empirical lookup / 128"), ("oracle_dynamics", 32, "Privileged true dynamics / 32"),
                               ("oracle_dynamics", 128, "Privileged true dynamics / 128"), ("optimal_oracle", None, "Privileged optimal oracle")):
        cells = [_cell(index.get((regime, "G", "core_family_macro", _series(method, budget)))) for regime in REGIMES]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines.extend(["", "## Per-instance learned goal control", ""])
    for regime in REGIMES:
        lines.extend([f"### {REGIME_NAMES[regime]}", "", "| Instance | GCBC | GCIQL | Dynamics / 32 | Dynamics / 128 |", "|---|---:|---:|---:|---:|"])
        for task, label in TASK_NAMES.items():
            cells = [_cell(index.get((regime, "G", f"instance:{task}", _series(method, budget)))) for method, budget, _ in LEARNED]
            lines.append(f"| {label} | {' | '.join(cells)} |")
        lines.append("")
    if figures:
        lines.extend(["![Per-instance G](figures/G_instances.png)", ""])
    lines.extend(["## Native rule and composition prediction", "",
                  "GCBC and GCIQL have no native R/C readout. Dynamics is shown once at its 32-query G condition; "
                  "R/C predictions themselves do not use a G search budget. Lookup and true dynamics are separately labeled references.", "",
                  "| Condition | Method/reference | R | C |", "|---|---|---:|---:|"])
    for regime in REGIMES:
        for method, label in (("dynamics", "Learned dynamics"), ("lookup", "Empirical lookup"), ("oracle_dynamics", "Privileged true dynamics")):
            cells = [_cell(index.get((regime, metric, "core_family_macro", _series(method, 32)))) for metric in ("R", "C")]
            lines.append(f"| {REGIME_NAMES[regime]} | {label} | {' | '.join(cells)} |")
    lines.extend(["", "## Structural-prior references", "",
                  "Constant XOR/position operators are fitted from training records under an explicit algebraic prior. "
                  "They are declared only for the instances below, with no all-core aggregate. "
                  "Deterministic references have three data-seed realizations, not three independent fits per seed.", "",
                  "| Condition | Instance / prior | R | C | G / 32 | G / 128 |", "|---|---|---:|---:|---:|---:|"])
    for regime in REGIMES:
        for task, method, label in (("grid-3x3", "fitted_xor", "Lights Out 3×3 / XOR"), ("grid-4x4", "fitted_xor", "Lights Out 4×4 / XOR"),
                                    ("pancake-6", "fitted_position", "Pancake 6 / position"), ("topspin-8-4", "fitted_position", "TopSpin 8/4 / position")):
            cells = [_cell(index.get((regime, metric, f"instance:{task}", _series(method, budget))))
                     for metric, budget in (("R", 32), ("C", 32), ("G", 32), ("G", 128))]
            lines.append(f"| {REGIME_NAMES[regime]} | {label} | {' | '.join(cells)} |")
    lines.extend(["", "## Coverage and prespecified comparisons", "",
                  "Updates are held fixed across nominal 25/50/100% coverage, not epochs or unique samples seen. "
                  "Primitive-query support and observed graph connectivity can change; `rows.json` preserves the "
                  "C diagnostics and G support for each actual selected view. Original G reachability labels "
                  "refer to the full corpus and remain separate.", "",
                  "| Condition | Contrast | Left − right | Interval |", "|---|---|---:|---:|"])
    for row in statistics["paired_contrasts"]:
        if row["scope"] == "core_family_macro" and row["metric"] == "G":
            a, b = row["left_minus_right"]
            lines.append(f"| {REGIME_NAMES[row['regime']]} | `{a}` − `{b}` | {row['mean']:.3f} | "
                         f"[{row['interval'][0]:.3f}, {row['interval'][1]:.3f}] |")
    if figures:
        lines.extend(["", "![Coverage effects](figures/G_coverage.png)"])
    lines.extend(["", "## Costs, failures and interpretation", "",
                  f"Recorded worker elapsed work across all retained phases/attempts: {costs['known_process_work_seconds']:.1f} seconds. "
                  f"Attempts with unknown duration: {costs['attempts_with_unknown_process_duration']}; "
                  f"unlinked attempt directories: {len(costs['unlinked_attempt_directories'])}. "
                  "This sum is neither experiment wall time nor CPU/GPU time. Shared fitting is counted once. "
                  "Prediction-archive reproduction is separate work recorded in its index.", "",
                  "See `rows.json` for failures, difficulty strata and supported diagnostics; `statistics.json` for all "
                  "seed values and contrasts; and `costs.json` for individual original/retry attempts. "
                  "The G normalized-cost companion uses the fixed instance horizon and penalizes all failures.", "",
                  "True dynamics with the matched bounded planner is a diagnostic comparison, not a guaranteed upper bound "
                  "on a learned model under that heuristic. The optimal oracle is privileged and has no comparable query budget. "
                  "These results do not establish unseen-rule, size, family or perception generalization."])
    if discussion:
        lines.extend(["", "[Evidence-based interpretation](DISCUSSION.md)"])
    return "\n".join(lines)+"\n"


def render_figures(statistics, output, *, final, synthetic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    output.mkdir()
    index = _index(statistics)
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    names = [name for _, _, name in LEARNED]
    prefix = "Synthetic rendering fixture: " if synthetic else "" if final else "Development qualification: "
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    def point(axis, x, row, color):
        if row is not None:
            lo, hi = row["interval"]
            axis.vlines(x, lo, hi, color=color, linewidth=1.5)
            axis.scatter([x], [row["mean"]], color=color, s=35, zorder=3)
    def configure(axis):
        axis.set_ylim(-.025, 1.025)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", color="#dddddd", linewidth=.7)
        axis.set_axisbelow(True)
        axis.set_ylabel("Goal success (equal-tier mean)")
    def save(fig, name, title):
        fig.suptitle(title+"\n95% paired hierarchical bootstrap; data/fit seed variation; fixed instances and cases", fontsize=11)
        fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(output / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, regime in zip(axes, REGIMES):
        for i, (method, budget, _) in enumerate(LEARNED):
            point(axis, i, index.get((regime, "G", "core_family_macro", _series(method, budget))), palette[i])
        axis.set_xticks(range(4), ["GCBC", "GCIQL", "Dynamics\n32 queries", "Dynamics\n128 queries"])
        axis.set_title(REGIME_NAMES[regime]); configure(axis)
    save(fig, "G_primary", prefix+"CoPuzzle v1: full-coverage family macro")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.8), constrained_layout=True)
    for axis, regime in zip(axes, REGIMES):
        for t, task in enumerate(TASK_NAMES):
            for i, (method, budget, _) in enumerate(LEARNED):
                point(axis, t+(i-1.5)*.16, index.get((regime, "G", f"instance:{task}", _series(method, budget))), palette[i])
        axis.set_xticks(range(len(TASK_NAMES)), list(TASK_NAMES.values()))
        axis.set_title(REGIME_NAMES[regime]); configure(axis)
    handles = [plt.Line2D([], [], color=color, marker="o", linestyle="", label=name) for color, name in zip(palette, names)]
    fig.legend(handles=handles, loc="outside lower center", ncols=4, frameon=False)
    save(fig, "G_instances", prefix+"CoPuzzle v1: full-coverage instances")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), constrained_layout=True)
    for axis, regime in zip(axes, REGIMES):
        for i, (method, budget, _) in enumerate(LEARNED):
            values = [(c, index.get((regime, "G", "core_family_macro", _series(method, budget, c)))) for c in (.25, .5, 1.0)]
            points = [(c, row) for c, row in values if row is not None]
            axis.plot([c for c, _ in points], [row["mean"] for _, row in points], color=palette[i], linewidth=1)
            for c, row in points:
                point(axis, c, row, palette[i])
        axis.set_xticks([.25, .5, 1], ["25%", "50%", "100%"])
        axis.set_xlabel("Nominal training-query coverage"); axis.set_title(REGIME_NAMES[regime]); configure(axis)
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False)
    save(fig, "G_coverage", prefix+"CoPuzzle v1: coverage at fixed update count")
    return matplotlib.__version__


def build_report(analysis, output, *, figures=False, discussion=None):
    analysis, output = Path(analysis), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest, statistics, rows, costs = load_analysis(analysis)
    output.mkdir(parents=True)
    plot_version = None
    if figures:
        plot_version = render_figures(statistics, output / "figures", final=manifest["role"] == "formal" and manifest["split"] == "test", synthetic=manifest.get("synthetic", False))
    if discussion is not None:
        shutil.copyfile(discussion, output / "DISCUSSION.md")
    (output / "REPORT.md").write_text(markdown(manifest, statistics, rows, costs, figures=figures, discussion=discussion is not None))
    (output / "report_source.py").write_bytes(Path(__file__).read_bytes())
    result = {"schema": "copuzzle.readable_report.v1", "analysis_manifest_sha256": _hash(analysis / "manifest.json"),
              "study_sha256": manifest["study_sha256"], "role": manifest["role"], "split": manifest["split"],
              "synthetic": manifest.get("synthetic", False),
              "interpretation_included": discussion is not None,
              "render_versions": {"python": platform.python_version(), "matplotlib": plot_version},
              "files": {p.relative_to(output).as_posix(): _hash(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def verify_report(directory, *, analysis_sha256, study_sha256):
    directory = Path(directory)
    manifest = _read(directory / "manifest.json")
    if (manifest.get("schema") != "copuzzle.readable_report.v1" or manifest.get("role") != "formal"
            or manifest.get("split") != "test" or manifest.get("synthetic") is not False
            or manifest.get("interpretation_included") is not True
            or manifest.get("analysis_manifest_sha256") != analysis_sha256
            or manifest.get("study_sha256") != study_sha256):
        raise ValueError("release report must contain interpretation of the verified formal analysis")
    if not {"REPORT.md", "DISCUSSION.md", "report_source.py"}.issubset(manifest["files"]):
        raise ValueError("readable report omits required evidence files")
    for name, digest in manifest["files"].items():
        if _hash(_inside(directory, name)) != digest:
            raise ValueError("readable report changed after rendering")
    if not (directory / "DISCUSSION.md").read_text().strip():
        raise ValueError("formal interpretation is empty")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--discussion", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_report(args.analysis, args.output, figures=args.figures, discussion=args.discussion), indent=2))


if __name__ == "__main__":
    main()
