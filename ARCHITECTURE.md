# Architecture (as-built)

This document describes how GeoNet is actually built as of `master` (October 2026). It covers the six-stage pipeline, the module dependency structure, the GRASS session machinery, and the numerics of stage 5. The algorithm mathematics live in [docs/algorithm_spec.md](docs/algorithm_spec.md); command-line usage in [docs/usage.md](docs/usage.md).

## Shape of the system

GeoNet is **not a package**. It is a flat tree of `pygeonet_*.py` scripts, each with a `main()` and an `if __name__ == '__main__'` guard, run in sequence from the repo root. There is no `setup.py`, `pyproject.toml`, console entry point, or installed API. State flows between stages exclusively through **GeoTIFF files** in the project's output directory.

Two upstream design decisions shape everything:

1. **`pygeonet_prepare.py` is a side-effectful import.** Every module starts with `import pygeonet_prepare as Parameters` — the import itself resolves the project `.cfg`, and creates the output directories. Its module-level globals (`Parameters.demFileName`, `Parameters.geonetResultsDir`, …) are the de-facto configuration object for the whole pipeline.
2. **Cross-module coupling uses star imports.** Stage scripts do `from pygeonet_rasterio import *`, `from pygeonet_plot import *`, and import `pygeonet_defaults as defaults`. This makes `defaults.<param>` the tuning surface and `Parameters.<path>` the path surface, but means module import order carries runtime meaning.

## The six-stage pipeline

```text
                     ┌────────────────────────────┐
   dem.tif ────────► │ Stage 1                    │ ───► PM_filtered_grassgis.tif
                     │ pygeonet_nonlinear_filter  │       (filtered DEM, Float32,
                     │ Perona-Malik diffusion     │        reprojected to EPSG)
                     └────────────────────────────┘
                                    │
                     ┌──────────────▼─────────────┐
                     │ Stage 2                    │ ───► <dem>_slope.tif
                     │ pygeonet_slope_curvature   │ ───► <dem>_curvature.tif
                     └──────────────┬─────────────┘
                                    │
                     ┌──────────────▼─────────────┐
                     │ Stage 3                    │  reads PM_filtered_grassgis.tif
                     │ pygeonet_grass_py3.py      │ ───► <dem>_fdr.tif     (flow dir, Int32)
                     │ GRASS r.watershed          │ ───► <dem>_fac.tif     (flow acc, Float64)
                     │ + r.stream.basins          │ ───► <dem>_outlets.tif (pits, Float32)
                     │                            │ ───► <dem>_basins.tif  (subbasins, Int16)
                     └──────────────┬─────────────┘
                                    │
                     ┌──────────────▼─────────────┐
                     │ Stage 4                    │  reads _fac.tif + _curvature.tif
                     │ pygeonet_skeleton_         │ ───► <dem>_flowskeleton.tif
                     │        definition          │ ───► <dem>_curvatureskeleton.tif
                     │                            │ ───► <dem>_skeleton.tif (dual)
                     └──────────────┬─────────────┘
                                    │
                     ┌──────────────▼─────────────┐
                     │ Stage 5                    │  reads _outlets/_basins/_curvature/
                     │ pygeonet_fast_marching     │        _fac/_skeleton .tif
                     │ numba start-point select   │ ───► <dem>_costfunction.tif
                     │ cost eval + skfmm travel   │ ───► <dem>_geodesicDistance.tif
                     └──────────────┬─────────────┘
                                    │
                     ┌──────────────▼─────────────┐
                     │ Stage 6                    │  reads _skeleton.tif +
                     │ pygeonet_channel_head_     │        _geodesicDistance.tif
                     │        definition          │ ───► <dem>_channelHeads.tif
                     │                            │ ───► <dem>_channelHeads.shp (OGR)
                     └────────────────────────────┘
```

Set-up scripts before the stages:

- `pygeonet_configure.py` — argparse CLI that writes `GeoNet_<project>.cfg` (five keys: `geofloodhomedir`, `projectname`, `dem_name`, `Input_dir`, `Output_dir`) and, when the home directory differs from the checkout, a `project_pointer.cfg` next to the scripts.
- `pygeonet_prepare.py` — on **import**: reads `project_pointer.cfg` (or globs `GeoNet*.cfg` next to the scripts, taking `glob(...)[0]`), parses the five keys, creates `GeoInputs/GIS/<project>`, `GeoOutputs/GIS/<project>`, `GeoOutputs/Inundation/<project>`, and publishes every downstream path as module globals. A missing cfg surfaces here as a bare `IndexError` from the glob.
- `pygeonet_check_setup.py` — stdlib-only pre-flight that mirrors prepare's cfg resolution, validates the five keys, the home dir, and the input DEM, and (with `--imports`) probes all third-party imports. No import side effects.
- `pygeonet_envinfo.py` — stdlib-only environment report (Python, platform, git commit, dependency versions) for bug reports; never crashes on missing deps.

## Module dependency map

```text
pygeonet_prepare (import-time side effects; shared globals = "Parameters")
    ▲ imported by every module below
pygeonet_defaults ("defaults"; tuning parameters)
    ▲ imported by stage scripts
pygeonet_rasterio (GDAL raster I/O)   pygeonet_plot (matplotlib)
    ▲                                      ▲
    ├──────────┬───────────┬───────────────┤
    │          │           │               │
 stage 1    stage 2     stage 4         stage 5/6
pygeonet_grass_py3 ──► pygeonet_grass_session (GRASS init dispatch)
pygeonet_vectorio (OGR shapefile writers) ◄── stage 6
```

Module-by-module:

| Module | Responsibility | Notes |
|---|---|---|
| `pygeonet_prepare` | cfg resolution + directory creation + shared path globals | import-time side effects; see above |
| `pygeonet_defaults` | every tunable parameter + reporting/plot toggles | the tuning surface; catalog in [docs/algorithm_spec.md](docs/algorithm_spec.md) |
| `pygeonet_rasterio` | GDAL reads/writes; publishes `geotransform`, `demPixelScale`, WKT into `Parameters` | `write_geotif_filteredDEM` reprojects to the DEM's EPSG authority code |
| `pygeonet_plot` | matplotlib raster/point plots guarded by `defaults.doPlot` | plotting is mostly commented out in stage `main()`s |
| `pygeonet_vectorio` | OGR shapefile writers | only `write_drainage_nodes` is wired into the pipeline (stage 6). `write_drainage_paths`, `write_cross_sections`, `write_bank_lines` retain Python-2 `xrange` loops and are not called by any stage — legacy GeoFlood-heritage code, do not import on Python 3 without fixing |
| `pygeonet_grass_session` | GRASS version detection + session init dispatch | 2026 addition; clean, documented, unit-tested |
| `pygeonet_check_setup`, `pygeonet_envinfo` | pre-flight + env report | 2026 additions; stdlib-only, no import side effects |
| `pygeonet_fast_marching` | stage-5 numerics + `FastMarchingDomainError` guard | 2026 hardening; see below |

Documentation quality is bimodal by lineage: the 2026 Kestrel additions (`grass_session`, `fast_marching` guard, `check_setup`, `envinfo`) have real docstrings; the upstream 2020 core (stages 1–4, rasterio, vectorio, plot, defaults) is essentially undocumented.

## GRASS session dispatch (stage 3)

GRASS GIS 7.8 and 8.x expose incompatible session-init signatures:

```text
GRASS 7.8: gsetup.init(gisbase, gisdb, location, mapset)
GRASS 8.x: gsetup.init(path, location=None, mapset=None, *, grass_path=None)
           # path = GISDBASE; installation dir moved to keyword grass_path
```

`pygeonet_grass_py3.py` therefore:

1. **Finds the start script** — Linux: probes `grass`, `grass84`, `grass78` with `--config path` (first success wins, `OSError` if none); Windows: `C:\OSGeo4W64\bin\grass78.bat`; macOS: `/Applications/GRASS/GRASS-7.8.app/`.
2. **Detects the major version** via `grass --config version`, parsed by `pygeonet_grass_session.parse_grass_major_version`; anything other than 7 or 8 exits with an error.
3. **Creates the `geonet` location** — GRASS 8: `g.create_location` from the georeferenced filtered DEM (GRASS 8's `g.proj` can no longer create a location); GRASS 7: `g.proj georef=`.
4. **Initializes the session** through `pygeonet_grass_session.init_grass_session(gsetup, …, grass_major)`, which picks the right call shape and raises `RuntimeError` on anything else. The script calls it at three points (after location creation, after mapset switch, after mapset creation) because the GRASS 8 signature requires an existing location/mapset.

The GRASS database lives at `~/grassdata` (`~\Documents\grassdata` on Windows). Stage 3 **deletes and recreates** the `geonet` location on every run, and (re)installs the `r.stream.basins` / `r.stream.watersheds` addons via `g.extension` — network and compiler required at runtime.

## Stage-5 design (the numerically hard stage)

Stage 5 computes, per qualifying subbasin, the fast-marching travel time from the basin's outlet across the channel network — the geodesic distance field stage 6 searches for channel heads.

1. **Start-point selection is a two-pass numba kernel.** `Fast_Marching_Start_Point_Identification` is `@njit(parallel=True)`: pass 1 iterates basin labels in `prange` and each label writes only its own slot of a `keep` mask (disjoint memory, race-free); pass 2 (`fmm_list_creation`) is a deterministic serial reduction gathering qualifying start points in label order. A basin qualifies when it covers more than `thresholdPercentAreaForDelineation` (0.1%) of the DEM's pixels and has more than `numBasinsElements` (2) elements. This fixed the original version's race on shared Python lists inside `prange` (nondeterministic order, lost appends, heap corruption) — PR #2.
2. **Cost evaluation.** `Local_Cost_Computation` evaluates `defaults.reciprocalLocalCostFn` via `eval()` with four names in scope (`flowArray`, `flowMean`, `skeletonFromFlowAndCurvatureArray`, `curvatureDemArray`); the fuller name list in the `pygeonet_defaults` comments is historical. Curvature is normalized to [0, 1] first (`doNormalizeCurvature=1`) and NaNs zeroed. Result written as `<dem>_costfunction.tif`.
3. **Per-basin travel-time solve.** For each start point: mask the basin, set `phi=1` where flow accumulation ≠ 0, `phi=-1` at the start point, `speed = cost`, then `skfmm.travel_time(phi, speed, dx=0.01)` — **wrapped in `travel_time_with_domain_guard`**, which re-raises scikit-fmm's opaque failures (ValueError/RuntimeError, notably the pre-2021.1.20 "Negative discriminant in time marcher quadratic" abort on rough cost fields, scikit-fmm issue #18) as `FastMarchingDomainError` carrying the catchment index and start point — PR #3. Each solved basin is scattered into the full-raster `geodesicDistanceArray` **inside** the loop — the per-basin scatter was lost in a refactor and restored by PR #4 (without it, only the final basin's distances survived). `∞` becomes NaN at the end; result written as `<dem>_geodesicDistance.tif`.
4. **Parallelism.** numba `prange` parallelizes the qualification pass; the travel-time loop itself is serial in-process. `dask.distributed.Client` is imported but not used on the current path — a hook from earlier per-basin scatter designs; `psutil` prints RAM usage per solve. Expect minutes on a ~0.7 GB DEM (the module's own banner says 5–7 min).

## Known incompatibilities

| Where | Trigger | Failure |
|---|---|---|
| `pygeonet_grass_py2.py` | any Python 3 | the Python-2-only GRASS 7.6 wrapper (`ConfigParser`, `grass.script`) cannot import on Python 3 — by design, not a defect |
| `pygeonet_vectorio.py` | Python 3, if you call the legacy writers | `xrange` undefined in `write_drainage_paths` / `write_cross_sections` / `write_bank_lines` |

The former stage-4 (`np.warnings` AttributeError on NumPy ≥ 2) and stage-6 (`from time import clock` ImportError on Python ≥ 3.8) rows were fixed by PR #10 and are regression-guarded by the import-sweep test (`tests/test_import_smoke.py`), which imports every py3-reachable `pygeonet_*.py` module on Python 3.11 with heavy dependencies stubbed. The remaining rows are upstream-era defects in code paths the unit-test suite does not execute. Run the full pipeline under the pinned conda env (Python 3.7.7, NumPy 1.19); stages 4 and 6 additionally run on modern Python (≥ 3.11, NumPy ≥ 2).

## Legacy boundary

- **`Old_Version/`** — ten upstream scripts (flow accumulation, network delineation, hydro flattening, …) superseded by the numbered pipeline. Unsupported; kept for lineage.
- **`pygeonet_grass_py2.py`** — Python 2-only stage-3 wrapper (`ConfigParser`, `time.clock`, `print >>`) targeting GRASS 7.6. Unsupported; stage 3 is `pygeonet_grass_py3.py`.
- **`GeoFlood` naming** — config keys (`geofloodhomedir`), the prepare docstring (`GeoFlood_<project>.cfg`), and cfg-path variables carry the GeoFlood name: GeoNet and GeoFlood share a workflow origin. `pygeonet_configure.py` actually writes `GeoNet_<project>.cfg`.

## Tests

`tests/` is stdlib `unittest` (not pytest), run via `python -m unittest discover -s tests -v`:

| File | Covers |
|---|---|
| `test_grass_session_init.py` | GRASS 7.8/8.x `init` call-shape dispatch, `parse_grass_major_version` — mocks `grass.script.setup`, no GRASS install |
| `test_fast_marching.py` | domain guard + travel-time helper — stubs gdal/rasterio/matplotlib/numba/dask/psutil in `sys.modules` |
| `test_stage5_determinism.py`, `test_stage5_scatter.py` | numba kernel determinism and the per-basin scatter (PR #2/#4 regressions) |
| `test_setup_scripts.py` | `check_setup` cfg resolution/validation, `envinfo` report |

Design rule: tests need only numpy + scikit-fmm + stdlib. Heavy dependencies are stubbed at import time and GRASS is mocked, so the suite runs green in the plain-pip environment of [SETUP.md](SETUP.md) Path B — that is the environment CI runs (Python 3.11), with no GDAL or GRASS present.

## CI

A single workflow, `.github/workflows/ci.yml` (`ci`): on every push to `master`/`develop` and every pull request, Python 3.11 on ubuntu-latest, `pip install numpy numba scikit-fmm matplotlib dask distributed rasterio psutil`, then `python -m unittest discover -s tests -v`. No docs build, lint, or typecheck; the GRASS-dependent smoke path auto-skips. (For this repository, CI/CD is intentionally disabled at the organization level; local verification stands as the gate for changes.)
