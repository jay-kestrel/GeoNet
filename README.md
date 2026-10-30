# GeoNet

Geomorphic feature extraction from high resolution topography data: channel networks and channel heads from lidar DEMs, using Perona-Malik nonlinear diffusion, GRASS GIS flow routing, geodesic minimum-cost paths, and fast-marching travel time.

This is the Kestrel fork of [`passaH2O/GeoNet`](https://github.com/passaH2O/GeoNet) (upstream dormant since September 2020). The fork's changes are tracked in [CHANGELOG.md](CHANGELOG.md).

When using GeoNet, please cite the following papers:

Passalacqua, P., T. Do Trung, E. Foufoula-Georgiou, G. Sapiro, W. E. Dietrich (2010), A geometric framework for channel network extraction from lidar: Nonlinear diffusion and geodesic paths, Journal of Geophysical Research Earth Surface, 115, F01002, doi:10.1029/2009JF001254.

Sangireddy, H., R. A. Carothers, C.P. Stark, P. Passalacqua (2016), Controls of climate, topography, vegetation, and lithology on drainage density extracted from high resolution topography data, Journal of Hydrology, 537, 271-282, doi:10.1016/j.jhydrol.2016.02.051.

## Documentation

| Document | Contents |
|---|---|
| [SETUP.md](SETUP.md) | Environment setup: full-pipeline (conda + GRASS GIS) and test-only paths, Python-version matrix |
| [ARCHITECTURE.md](ARCHITECTURE.md) | As-built architecture: the six-stage pipeline, module coupling, GRASS session dispatch, stage-5 numerics |
| [docs/algorithm_spec.md](docs/algorithm_spec.md) | Technical spec: diffusion, routing, skeletonization, cost function grammar, fast marching, channel-head search, the full parameter catalog |
| [docs/usage.md](docs/usage.md) | Per-script CLI reference and the small importable helper surface |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Running the tests, test design patterns, commit and PR conventions |
| [CHANGELOG.md](CHANGELOG.md) | Changes on the fork, per merged PR |

## Environment

GeoNet is **not** an installable Python package — it is a flat tree of `pygeonet_*.py` scripts run in order from the repo root. There is no `setup.py`/`pyproject.toml`; do not run `python setup.py install`.

```sh
conda env create -f GeoNetEnv.yml
conda activate GeoNetEnv
```

`GeoNetEnv.yml` is a **2020-era win-64 conda environment** (Python 3.7.7, GDAL 2.3.3, NumPy 1.19.1) and installs scikit-fmm pinned at **`scikit-fmm==2021.7.8`** through its `pip:` section — there is no need to build scikit-fmm from source. The stale trailing `prefix:` line in the file is a leftover personal path and is ignored on create.

Two things the env file does **not** give you:

1. **GRASS GIS** — a separate system install (see below).
2. **Linux/macOS** — the env pins and the `requirements.txt` export are win-64 specific. See [SETUP.md](SETUP.md) for the platform story and a test-only path.

`requirements.txt` is **not** a pip requirements file: it is a `conda create --file` `@EXPLICIT` spec of win-64 package URLs. `pip install -r requirements.txt` fails.

### GRASS GIS

Stage 3 requires a GRASS GIS **system** install, separate from the conda env:

- Download the official release from <https://grass.osgeo.org/download/>.
- **GRASS 7.8 and 8.x are both supported** (8.x support was added by PR #1 via the version-dispatch in `pygeonet_grass_session.py`). Stage 3 probes `grass`, `grass84`, and `grass78` start scripts on Linux and detects the major version at runtime.
- GRASS 7.6 is not supported: its legacy `pygeonet_grass_py2.py` wrapper was Python 2 code and has been removed.

### Python floor — read before choosing an interpreter

There is no declared supported version; three different versions are in play:

| Environment | Python | Notes |
|---|---|---|
| `GeoNetEnv.yml` (full pipeline) | 3.7.7 | The version the 2020-era pins were built for |
| CI (`ci.yml` unit tests) | 3.11 | Tests only, not the pipeline stages |

Stage 6 runs on any Python 3 again (the unused `from time import clock` was removed in PR #10), and stage 4 runs on NumPy ≥ 2 (the `np.warnings.filterwarnings` call was removed). The CI import-sweep test (`tests/test_import_smoke.py`) imports every py3-reachable `pygeonet_*.py` module on Python 3.11 with heavy dependencies stubbed, so a modern-Python import regression fails the build. The conda env as pinned (Python 3.7.7) remains the only verified end-to-end pipeline baseline; further issues live in [ARCHITECTURE.md](ARCHITECTURE.md).

## Configuration and Setup

Create a configuration file for your project:

```sh
python pygeonet_configure.py
```

Optional arguments:

```text
-dir  </path/to/GeoNet/Home_Directory>   home directory for the cfg and input/output folders (default: the repo checkout)
-p    [projectName]                      project name (default: my_project)
-n    [DEM_Name]                         input DEM name without extension (default: dem)
--input_dir  [Input_Directory_Name]      (default: GeoInputs)
--output_dir [Output_Directory_Name]     (default: GeoOutputs)
```

This writes `GeoNet_<project>.cfg` into the home directory (plus a `project_pointer.cfg` next to the scripts when the home directory differs from the checkout). Then create the file structure:

```sh
python pygeonet_prepare.py
```

Default file structure:

```text
GeoNet/
├── GeoNet_<project>.cfg          ← configuration file
├── GeoInputs/
│   └── GIS/
│       └── <project name>/
│           └── dem.tif           ← your input DEM
└── GeoOutputs/
    └── GIS/
        └── <project name>/
```

> Note: the `pygeonet_prepare.py` docstring still calls this `GeoFlood_<project>.cfg` — GeoNet and GeoFlood share a workflow lineage, and some config paths in the code carry the `GeoFlood` name. The file `pygeonet_configure.py` actually writes is `GeoNet_<project>.cfg`.

### Pre-flight checks

Verify the configuration, home directory, input DEM, and project directories before running the pipeline stages:

```sh
python pygeonet_check_setup.py
```

Add `--imports` to also check that every Python dependency the pipeline needs can be imported.

Print Python, git, and dependency versions for bug reports:

```sh
python pygeonet_envinfo.py
```

Include the output of `pygeonet_envinfo.py` when reporting a problem to the maintainers (GitHub Issues are disabled on this fork; report via a pull request comment or your Kestrel contact).

## Scripts

Run the numbered stages in order from the repo root. Each stage reads and writes GeoTIFFs under the file structure above; see [docs/usage.md](docs/usage.md) for exact inputs and outputs.

1. Perona-Malik non-linear diffusion filter of the DEM:

   ```sh
   python pygeonet_nonlinear_filter.py
   ```

2. Slope and curvature of the filtered DEM:

   ```sh
   python pygeonet_slope_curvature.py
   ```

3. Flow direction, flow accumulation, outlets, and basins via GRASS GIS `r.watershed`:

   ```sh
   python pygeonet_grass_py3.py
   ```

   Works with GRASS GIS 7.8 and 8.x (detected at runtime). The `python pygeonet_grass_py2.py` wrapper shown by older READMEs was Python 2 code and has been removed.

4. Flow-accumulation and curvature skeleton:

   ```sh
   python pygeonet_skeleton_definition.py
   ```

5. Geodesic minimum-cost path and fast marching algorithm:

   ```sh
   python pygeonet_fast_marching.py
   ```

6. Channel head detection:

   ```sh
   python pygeonet_channel_head_definition.py
   ```

> Note: further research still needs to be done on the optimal threshold for identifying channel heads. Preliminary studies found a quantile of 0.3 on the skeleton-component histogram (hardcoded in `pygeonet_channel_head_definition.py`) to be sufficient, but this estimate can definitely be improved using analytical methods.
