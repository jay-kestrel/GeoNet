# Setting up GeoNet

GeoNet is run from a checkout — there is nothing to `pip install`. What you need depends on what you want to do:

- **Run the full pipeline** (stages 1–6 on real DEMs) → Path A: conda env + GRASS GIS.
- **Run the test suite / develop** → Path B: a plain pip environment, no GRASS, no conda.

There is no supported middle ground: the pipeline's heavy dependencies (GDAL, GRASS) and the test suite's stubbed environment are documented separately because they genuinely differ.

## The Python-version matrix first

No supported version is declared anywhere in the repo. Three versions are in play, and the pipeline is not consistent across them:

| Component | Python | Status |
|---|---|---|
| `GeoNetEnv.yml` full-pipeline env | 3.7.7 | What the 2020-era pins were built and verified for |
| CI (`ci.yml`) | 3.11 | Runs the **unit tests only** — never the pipeline stages |
| Stages 1–5 as scripts | 3.7.x–3.7 | Unverified above 3.7 for full runs; the CI import-sweep imports every stage module on 3.11 |
| Stage 6 (`pygeonet_channel_head_definition.py`) | 3.7.x–3.11+ | Fixed by PR #10 — the unused `from time import clock` was removed; imports on any Python 3 |

And one NumPy constraint:

| Component | NumPy | Status |
|---|---|---|
| Stage 4 (`pygeonet_skeleton_definition.py`) | ≥ 2.0 | Fixed by PR #10 — the `np.warnings.filterwarnings` call was removed |
| CI test env | whatever `pip install numpy` yields (currently 2.x on Python 3.11) | Fine for the tests, and the import-sweep exercises stage 4's module import |

Both former breakages were **fixed by PR #10** and are regression-guarded by the CI import-sweep test (`tests/test_import_smoke.py`). The pinned Python 3.7.7 env (Path A) remains the only verified end-to-end baseline.

## Path A — full pipeline (conda env + GRASS GIS)

> **Honest framing:** this path reproduces the environment the upstream project used around 2020 on **Windows 64-bit**. The pins are old, the file is win-64 specific, and it has not been modernized. It is still the only environment every stage is known to work in.

### 1. Create the conda environment

```sh
conda env create -f GeoNetEnv.yml
conda activate GeoNetEnv
```

What you get (key pins): Python 3.7.7, numpy 1.19.1, numba 0.50.1, gdal 2.3.3, rasterio 1.0.21, scipy 1.5.0, dask/distributed 2.23, matplotlib 3.3.1, statsmodels 0.11.1, psutil 5.7.2 — plus scikit-fmm **2021.7.8** from the env file's `pip:` section.

Notes:

- The `pip:` pin `scikit-fmm==2021.7.8` is deliberate: scikit-fmm ≥ 2021.1.20 fixes the "Negative discriminant in time marcher quadratic" abort (scikit-fmm issue #18) that kills stage 5 on the older 2019.1.30 release, and 2021.7.8 is verified to build and run on Python 3.7. **Do not** replace this with the old build-from-source instructions — they install the 2019 version and reintroduce the abort.
- The trailing `prefix: C:\Users\alecc\Anaconda3\envs\GeoNetEnv` line is a stale personal path written by `conda env export`; conda ignores it on create. It can be deleted.
- The `channels:`/`dependencies:` pins are win-64 build strings. On Linux/macOS, conda will either fail to solve or produce a different-but-unverified stack — see "Other platforms" below.

### 2. Install GRASS GIS (system-wide, not from conda)

The env file does **not** contain GRASS GIS. Stage 3 shells out to a GRASS start script and needs:

- **GRASS GIS 7.8 or 8.x** (both supported — `pygeonet_grass_session.py` dispatches on the major version because their `grass.script.setup.init` signatures are incompatible).
- Install from <https://grass.osgeo.org/download/>, or via your OS package manager (`apt install grass`, `conda install -c conda-forge grass` in a *separate* env, etc.).

Stage 3 looks for the GRASS start script:

| Platform | Probed |
|---|---|
| Linux | `grass`, `grass84`, `grass78` (first one that answers `--config path`) |
| Windows | `C:\OSGeo4W64\bin\grass78.bat` |
| macOS | `/Applications/GRASS/GRASS-7.8.app/` |

On Windows/macOS the path is hardcoded in `pygeonet_grass_py3.py` — adjust it there if your install differs. GRASS 7.6 works only with the legacy `pygeonet_grass_py2.py` (Python 2, unsupported).

### 3. Runtime requirements of stage 3

On every run, stage 3 (re)installs the GRASS addons `r.stream.basins` and `r.stream.watersheds` with `g.extension` (removing them first if present). This means stage 3 needs **network access and a C compiler** at runtime, and the first run is slow. This is a documented gotcha, not a bug.

### 4. Configure and run

Follow the [README](README.md#configuration-and-setup): `pygeonet_configure.py` → place your DEM → `pygeonet_prepare.py` → `pygeonet_check_setup.py` → stages 1–6.

## Path B — test-only environment (what CI runs)

The unit tests never import GDAL or GRASS: they stub heavy modules in `sys.modules` and mock `grass.script.setup`. This is exactly the CI set from `.github/workflows/ci.yml`:

```sh
python -m venv .venv && source .venv/bin/activate   # any Python 3.8+; CI uses 3.11
pip install numpy numba scikit-fmm matplotlib dask distributed rasterio psutil
python -m unittest discover -s tests -v
```

No GDAL, no statsmodels, no GRASS. The full test suite passes in this environment by design — see [CONTRIBUTING.md](CONTRIBUTING.md) for the stubbing pattern the tests use and the rules for keeping it that way.

## About `requirements.txt`

`requirements.txt` is **not** a pip requirements file, despite the name:

```text
# platform: win-64
@EXPLICIT
https://conda.anaconda.org/conda-forge/win-64/git-2.28.0-1.tar.bz2
...
```

It is a conda **`@EXPLICIT`** export: 130 lines of win-64 package tarball URLs, produced by `conda list --explicit`. Use it only as its header says:

```sh
conda create --name <env> --file requirements.txt   # win-64 only
```

`pip install -r requirements.txt` fails (pip cannot parse `@EXPLICIT` specs or install `.tar.bz2` conda packages). For anything pip-based, use Path B's package list, or the conda env from Path A.

## Other platforms (Linux/macOS)

There is no maintained Linux/macOS pipeline environment. The blockers, in order of friction:

1. **GDAL 2.3.3** pins are win-64 build strings; a modern conda-forge GDAL on Linux is unverified with this code.
2. **GRASS start-script discovery** works on Linux (probed at runtime), but Windows/macOS paths are hardcoded in `pygeonet_grass_py3.py`.
3. The full pipeline is otherwise platform-agnostic NumPy/SciPy code.

If you attempt Linux/macOS, report your results to the maintainers with `python pygeonet_envinfo.py` output. The test suite (Path B) is the only officially cross-platform surface.
