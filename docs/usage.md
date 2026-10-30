# Using GeoNet — script CLIs and helper surface

GeoNet has **no package API**: it is not installable, and there is nothing to `import geonet`. The user-facing interface is (1) the numbered script pipeline run from the repo root and (2) a small set of documented helper functions added on the Kestrel fork. This document covers exactly those — no API is invented.

Setup lives in [SETUP.md](../SETUP.md); the algorithm behind each script in [algorithm_spec.md](algorithm_spec.md).

## The working pattern

```sh
python pygeonet_configure.py -dir /home/me/geonet -p my_project -n mydem
# put the DEM at /home/me/geonet/GeoInputs/GIS/my_project/mydem.tif
python pygeonet_prepare.py
python pygeonet_check_setup.py --imports
python pygeonet_nonlinear_filter.py
python pygeonet_slope_curvature.py
python pygeonet_grass_py3.py            # needs GRASS GIS 7.8/8.x installed
python pygeonet_skeleton_definition.py
python pygeonet_fast_marching.py
python pygeonet_channel_head_definition.py   # needs Python 3.7.x — see Known limitations
```

Every script reads the same project cfg **via the import of `pygeonet_prepare`** — there are no per-stage flags. All outputs land in `GeoOutputs/GIS/<project>/`.

## Setup scripts

### `pygeonet_configure.py`

Writes `GeoNet_<project>.cfg` into the home directory and a `project_pointer.cfg` next to the scripts (when home ≠ checkout).

| Flag | Default | Meaning |
|---|---|---|
| `-dir`, `--GeoNethomedir` | the repo checkout | Home directory holding the cfg and input/output folders |
| `-p`, `--project` | `my_project` | Project name; name of the folders inside GIS directories |
| `-n`, `--DEM_name` | `dem` | Input DEM name (no extension); prefix for all outputs |
| `--input_dir` | `GeoInputs` | Inputs folder name |
| `--output_dir` | `GeoOutputs` | Outputs folder name |

### `pygeonet_prepare.py`

No flags. Imports resolve the cfg (pointer file first, then `GeoNet*.cfg` glob) and create the directory tree. **Running it a second time is harmless** (directory creation is `if not exists`), but note the side effects happen at *import*, so any script that imports `pygeonet_prepare` has already done this work.

### `pygeonet_check_setup.py`

Pre-flight validation — cfg resolution, five cfg keys, home directory, input DEM; project directories warn if missing (prepare creates them). Exit code 1 on any failure.

| Flag | Effect |
|---|---|
| `--imports` | Additionally probe every third-party module the pipeline imports (`numpy`, `scipy`, `matplotlib`, `osgeo.gdal`, `rasterio`, `numba`, `skfmm`, `psutil`, `dask.distributed`, `statsmodels`, `grass.script`) and fail on the ones that cannot be imported, with the real import error |

### `pygeonet_envinfo.py`

No flags. Prints Python version, platform, git commit, and the installed version of every pipeline dependency (missing modules report "not installed" and never crash the report). Always exits 0. Attach the output when reporting problems to the maintainers — Issues are disabled on this fork.

## Pipeline stages

| Stage | Command | Reads | Writes |
|---|---|---|---|
| 1 | `python pygeonet_nonlinear_filter.py` | input DEM | `PM_filtered_grassgis.tif` |
| 2 | `python pygeonet_slope_curvature.py` | filtered DEM | `<dem>_slope.tif`, `<dem>_curvature.tif` |
| 3 | `python pygeonet_grass_py3.py` | filtered DEM | `<dem>_fdr.tif`, `<dem>_fac.tif`, `<dem>_outlets.tif`, `<dem>_basins.tif` |
| 4 | `python pygeonet_skeleton_definition.py` | `_fac`, `_curvature` | `<dem>_flowskeleton.tif`, `<dem>_curvatureskeleton.tif`, `<dem>_skeleton.tif` |
| 5 | `python pygeonet_fast_marching.py` | `_outlets`, `_basins`, `_curvature`, `_fac`, `_skeleton`, filtered DEM | `<dem>_costfunction.tif`, `<dem>_geodesicDistance.tif` |
| 6 | `python pygeonet_channel_head_definition.py` | `_skeleton`, `_geodesicDistance` | `<dem>_channelHeads.tif`, `<dem>_channelHeads.shp` |

Stage 3 notes: needs a GRASS GIS 7.8/8.x system install; deletes and recreates the `geonet` GRASS location each run; (re)installs the `r.stream.basins`/`r.stream.watersheds` addons via `g.extension` (network + compiler required at runtime).

## Importable helper surface (Kestrel additions)

Four modules are safe to import and expose documented functions. Everything else in the repo (`pygeonet_prepare` especially — see the import-time side effects in [ARCHITECTURE.md](../ARCHITECTURE.md#shape-of-the-system)) is pipeline plumbing, not API.

### `pygeonet_grass_session`

GRASS session initialization shared by stage 3.

| Function | Signature | Behavior |
|---|---|---|
| `parse_grass_major_version` | `(version_string) -> int or None` | Parse `grass --config version` output (e.g. `"8.4.1"`, `"GRASS GIS 7.8.8"`) to major int; `None` if unparsable |
| `init_grass_session` | `(gsetup, gisbase, gisdb, location, mapset, grass_major) -> session` | Call `gsetup.init` with the right signature for major version 7 (`(gisbase, gisdb, location, mapset)`) or 8 (`(gisdb, location, mapset, grass_path=gisbase)`); `RuntimeError` on anything else |

### `pygeonet_fast_marching`

| Name | Signature | Behavior |
|---|---|---|
| `FastMarchingDomainError` | `(RuntimeError)` | Raised when the fast-marching travel-time solve fails for a catchment; carries catchment index and start point |
| `travel_time_with_domain_guard` | `(phi, speed, dx, catchmentIndex, startPoint) -> array` | `skfmm.travel_time(phi, speed, dx=dx)` with `ValueError`/`RuntimeError` re-raised as `FastMarchingDomainError` with context |

### `pygeonet_check_setup`

Stdlib-only, no import side effects.

| Function | Signature | Behavior |
|---|---|---|
| `resolve_cfg_path` | `(script_dir) -> (path, None) or (None, reason)` | Resolve the project cfg exactly as `pygeonet_prepare` does at import (pointer file, then sorted `GeoNet*.cfg` glob) |
| `read_cfg` | `(cfg_path) -> dict` | Parse and validate the five cfg keys; `SetupError` with a readable message on problems |
| `expected_layout` | `(cfg) -> dict` | Paths the pipeline expects (home dir, input DEM, project dirs) with kinds `dir`/`file`/`dir-warn` |
| `run_checks` | `(cfg) -> [(level, label, detail)]` | Check every expected path; levels `ok`/`warn`/`fail` |
| `probe_imports` | `([names]) -> [(name, imported, detail)]` | Try importing each module; the exception text is the diagnostic |
| `main` | `(argv=None, script_dir=None, probe_names=PIPELINE_IMPORTS) -> int` | Run all checks; 0 when clean, 1 otherwise |

### `pygeonet_envinfo`

Stdlib-only, no import side effects.

| Function | Signature | Behavior |
|---|---|---|
| `collect_env_info` | `(repo_dir=None, dependencies=DEPENDENCIES) -> dict` | Python/platform/git-commit/dependency versions |
| `format_env_info` | `(info) -> str` | Render the dict as pasteable text |
| `dependency_version` | `(name) -> str` | Installed version or `'not installed'`; never raises |
| `git_commit` | `(repo_dir) -> str or None` | Short HEAD of the containing git repo, or `None` |
| `main` | `(repo_dir=None, dependencies=None) -> int` | Print the report; always returns 0 |

## Importing pipeline modules directly (discouraged but real)

If a script imports `pygeonet_prepare`, the cfg is resolved and directories are created **at import time**; a missing cfg raises a bare `IndexError` there. Modules also mutate `Parameters` globals as they run (`pygeonet_rasterio` sets `geotransform`/`demPixelScale`/`inputwktInfo` on first DEM read). Star imports (`from pygeonet_rasterio import *`) mean stage scripts see rasterio and plot helpers directly.

## Known limitations when running

- **Stage 6 fails on Python ≥ 3.8** (`from time import clock`). Use the pinned conda env (Python 3.7.7) for a full pipeline; see [SETUP.md](../SETUP.md).
- **Stage 4 fails on NumPy ≥ 2.0** (`np.warnings`).
- **No cross-platform pipeline env exists.** The conda pins and `requirements.txt` are win-64; `requirements.txt` is a conda `@EXPLICIT` file, not pip.
- **`Old_Version/` and `pygeonet_grass_py2.py` are legacy and unsupported** — do not run them.
- The channel-network / cross-section / bank-line shapefile writers in `pygeonet_vectorio` are GeoFlood-heritage code with Python-2 `xrange` loops; they are not called by the pipeline. Only the channel-heads writer is wired in (stage 6).
