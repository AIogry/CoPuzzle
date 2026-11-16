# Contributing to CoPuzzle

Keep environment semantics, data contracts and algorithm experiments separate.
Public API and protocol documentation belongs in root Markdown files; `docs/`
is ignored local research material, and `artifacts/` contains generated data.
Runtime code and required tests must not depend on either ignored directory.

## Validation

Install the project into a dedicated environment with `.[dev,render]` to run all
physical tests. On a headless Linux host with EGL support:

```sh
MUJOCO_GL=egl python -m pytest -q
python -m build --wheel --no-isolation
```

Run focused tests while developing. Before committing a new cross-family
protocol, run the full suite and verify at least one real saved-action replay
per affected physical family. Check a wheel from outside the source checkout.
Tests requiring optional local archives must explicitly skip when those archives
are absent; the functional API must still work in a clean installation.

## Data and compatibility

Do not overwrite validated artifacts, relabel a historical facility as the
current one, or silently change a trajectory schema. Preserve original actions,
observations, target semantics, failure outcomes and configuration/source hashes.
A new continuing-play or hindsight-label contract belongs in an explicitly
versioned interface. Goal changes, physical resets, timeouts and collector
cutoffs have distinct meanings. Reference plans and controller phase labels are
privileged metadata and must not leak into learner inputs.

## Commits and publication

Commit coherent, validated changes with the concrete behavior and relevant test
results in the message. Use `git diff --check` and inspect staged paths before
committing. Respect `.gitignore`; generated datasets/videos and local notes are
not part of a source commit. Use regular fast-forward pushes and verify the
remote ref; never overwrite remote history to bypass divergence. Keep public
changelog entries under Unreleased until an actual package release is prepared.
