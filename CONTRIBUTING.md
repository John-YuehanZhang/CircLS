# Contributing

## Setup

```bash
git clone https://github.com/John-YuehanZhang/CircLS.git
cd CircLS
python3.12 -m venv venv && source venv/bin/activate   # python 3.10-3.12
pip install -e '.[test]'
```

## Tests

```bash
python -m pytest tests -q               # compiler suite
python -m pytest lightstim/tests -q     # vendored LightStim suite
```

Both suites must pass before a PR. Tests that need optional pieces
skip themselves: the BP+OSD CLI tests skip without `stimbposd`, GPU
tests skip unless `RUN_GPU_TESTS=1` is set on a CUDA machine, the
mixed-GHZ benchmark tests skip without a TopoLS checkout
(`TOPOLS_DIR`), and the Gidney cross-checks skip unless
`GIDNEY_SRC` points at the zenodo 7487893 `code/src` directory.

The paper harnesses in `experiments/` are not part of the test
gate; they expect external baseline checkouts (`TOPOLS_DIR`,
`QASMBENCH_ROOT`) and are kept for reproducibility.

## Extending

- **A pipeline decision** (placement, routing, ordering,
  re-selection, lifetimes): don't fork the stage — use the hooks in
  `docs/API_HOOKS.md`. If a hook's contract is too narrow for your
  case, open an issue describing the decision you want to inject.
- **The construction rules** (`circls/core/multi_patch_coupler.py`,
  `circls/core/joint_merge.py`) are correctness-critical; changes there
  need new rule-level tests alongside the existing seam-table cases
  in `tests/`.
- **Vendored LightStim** (`lightstim/`): fixes welcome, but general
  improvements should also be offered upstream
  (<https://github.com/QuTone/LightStim>); see `VENDORED.md`.

## Pull requests

- Keep commits scoped; explain *why* in the message body when the
  change is not obvious.
- New behavior needs a test that fails without the change.
- CI runs both suites on Python 3.11; make sure `pip install -e
  '.[test]'` from a clean environment is all a reviewer needs.
- Contributions are accepted under the repository's Apache-2.0
  license (Apache-2.0 section 5).
