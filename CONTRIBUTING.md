# Contributing to GeoNet

Thanks for contributing to the Kestrel fork of GeoNet. This repo is a script-based research pipeline, not a package — the contribution workflow is shaped around that.

## Dev environment

Use the **test-only environment** from [SETUP.md](SETUP.md#path-b--test-only-environment-what-ci-runs) (plain pip, Python 3.11): the unit tests stub GDAL/GRASS, so you do not need conda, GDAL, or a GRASS install to develop. Full-pipeline testing needs the Path A conda env plus GRASS GIS — only necessary when you are changing a stage's runtime behavior on real DEMs.

```sh
python -m venv .venv && source .venv/bin/activate
pip install numpy numba scikit-fmm matplotlib dask distributed rasterio psutil
python -m unittest discover -s tests -v
```

## Repo layout you will touch

```text
pygeonet_*.py            flat script modules (no package; run from repo root)
tests/                   stdlib unittest suite
docs/                    markdown docs (algorithm spec, usage)
GeoNetEnv.yml            2020-era win-64 conda env (do not modernize casually)
requirements.txt         conda @EXPLICIT win-64 export — NOT pip-installable
Old_Version/             legacy upstream scripts — do not modify or run
```

There is no `pyproject.toml`, no install step, no lint/typecheck config. Markdown and code style is whatever the file you are editing already does.

## Running and writing tests

The suite is **stdlib `unittest`** (not pytest):

```sh
python -m unittest discover -s tests -v                # everything
python -m unittest tests.test_grass_session_init -v    # one module
```

The five test files and what they cover are tabled in [ARCHITECTURE.md](ARCHITECTURE.md#tests).

### The heavy-dep stubbing pattern — keep it working

The suite must run with **only numpy, scikit-fmm, and the stdlib** installed. Tests that need pipeline modules stub every heavy third-party dependency in `sys.modules` **before** importing the module under test:

```python
for name in ("osgeo", "osgeo.gdal", "osgeo.osr", "osgeo.ogr", "rasterio", ...):
    sys.modules[name] = types.ModuleType(name)   # or a stub object
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pygeonet_...
```

Rules this imposes on contributions:

- **Never add a module-level heavy import to a module the tests import** without also adding a stub to the relevant test files. `check_setup`, `envinfo`, `grass_session`, and the fast-marching guard are import-safe today; the import-sweep test (`tests/test_import_smoke.py`) enforces it for every py3-reachable `pygeonet_*.py` module with heavy dependencies stubbed — keep new code that way when it is reasonable.
- **GRASS is always mocked.** Tests never require a GRASS install; `test_grass_session_init.py` stands in for `grass.script.setup` with a recording fake. Any new GRASS-coupled logic should be testable through an injected `gsetup`-like object, not by calling GRASS.
- **Prefer testing pure helpers** (version parsing, cfg resolution, numba kernels on synthetic arrays) over end-to-end script runs.

## Former breakages — resolved by PR #10, guarded by the import sweep

- Stage 6 (`pygeonet_channel_head_definition.py`) no longer imports `time.clock` (removed in PR #10); it imports on any Python 3.
- Stage 4 (`pygeonet_skeleton_definition.py`) no longer uses `np.warnings` (stdlib `warnings` since PR #10); its threshold path runs on modern NumPy.
- `pygeonet_vectorio.py`'s network/cross-section/bank-line writers no longer use `xrange` (PR #10); they remain uncalled by the pipeline.

Behavior-changing fixes like these are **ordinary code PRs** (they get real review and a regression test each); `tests/test_import_smoke.py` now imports every `pygeonet_*.py` module on Python 3 and pins the stage-4 threshold path, so a regression in these areas fails the suite.

## Commit and PR conventions

- **Conventional Commits**, with the scopes the repo already uses — `fix(grass):`, `fix(numba):`, `fix(fmm):`, `feat(cli):`, `docs:`. Example: `docs: add architecture and algorithm specification`.
- **PRs are squash-merged** into `master` (the history is linear; PR titles carry the `(#N)` suffix).
- PRs target `master`. **GitHub Issues are disabled on this fork** — propose changes in a PR description or a discussion with the maintainers, not an issue.
- Fork posture: upstream `passaH2O/GeoNet` is dormant (last push September 2020); do not expect or wait for upstream syncs.
- Keep diffs focused: when the test workflow runs it has no path filters, so every push runs the whole test job — small PRs keep that cheap. (GitHub Actions are currently disabled for jay-kestrel repositories at the organization level; local test runs stand in as the gate.)

## Bug reports

Include the output of:

```sh
python pygeonet_envinfo.py
```

(Python, platform, git commit, dependency versions) along with the failing stage, the command line, and the cfg — the repo has no version string anywhere, so this context is what makes reports actionable. Since Issues are disabled on the fork, deliver reports via PR comments or your Kestrel contact.
