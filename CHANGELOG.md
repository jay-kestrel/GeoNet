# Changelog

All notable changes to GeoNet are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). GeoNet itself has **no version string or release process** — this changelog is keyed to the Kestrel fork's pull-request history. The upstream project (`passaH2O/GeoNet`) has been dormant since September 2020; everything below "Kestrel fork" is fork work on `master`.

## Kestrel fork

### Unreleased

#### Added

- `docs:` comprehensive documentation refresh — README accuracy pass (stage-3 command fix, GRASS 8.x, pinned scikit-fmm install, Python-floor statement, GRASS GIS install pointer), `SETUP.md` (full-pipeline vs test-only environments, `requirements.txt` is a conda `@EXPLICIT` spec), `ARCHITECTURE.md` (six-stage pipeline, module coupling, GRASS session dispatch, stage-5 numerics, legacy boundaries), `docs/algorithm_spec.md` (algorithm and full `pygeonet_defaults` parameter catalog), `docs/usage.md` (per-script CLIs and the importable helper surface), `CONTRIBUTING.md`, and this changelog.

#### Fixed

- **GRASS 8.x support** ([#1](https://github.com/jay-kestrel/GeoNet/pull/1)): stage 3 now supports GRASS GIS 8.x alongside 7.8 — `pygeonet_grass_session.py` detects the major version and dispatches between the incompatible `gsetup.init` signatures; GRASS 8 location creation uses `g.create_location` from the filtered DEM.
- **Deterministic, crash-free stage-5 start-point selection** ([#2](https://github.com/jay-kestrel/GeoNet/pull/2)): `Fast_Marching_Start_Point_Identification` no longer appends to shared Python lists inside `prange` (which raced: nondeterministic order, lost appends, SIGSEGV/SIGABRT). A parallel pass writes a per-label keep-mask; a serial reduction gathers start points in label order.
- **Fast-marching domain guard** ([#3](https://github.com/jay-kestrel/GeoNet/pull/3)): opaque scikit-fmm failures — including the "Negative discriminant in time marcher quadratic" abort on scikit-fmm < 2021.1.20 — are re-raised as `FastMarchingDomainError` with the catchment index and start point attached.
- **Per-basin geodesic distance scatter restored** ([#4](https://github.com/jay-kestrel/GeoNet/pull/4)): PR #3's refactor left only the final basin's travel times in the full-raster `geodesicDistanceArray`; every basin is now scattered into the raster inside the solve loop again.
- **Setup pre-flight and environment info** ([#5](https://github.com/jay-kestrel/GeoNet/pull/5)): `pygeonet_check_setup.py` validates the cfg (exactly as `pygeonet_prepare` resolves it at import), home directory, input DEM, and project directories, and probes third-party imports with `--imports`; `pygeonet_envinfo.py` prints a stdlib-only Python/git/dependency report for bug reports.
- **Modern-Python compatibility for stages 4 and 6** ([#10](https://github.com/jay-kestrel/GeoNet/pull/10)): removed the unused `from time import clock` (stage 6 was unimportable on Python ≥ 3.8) and the `np.warnings.filterwarnings` call (stage 4 raised AttributeError on NumPy ≥ 2); a new import-sweep test (`tests/test_import_smoke.py`) imports every py3-reachable `pygeonet_*.py` module with heavy dependencies stubbed.

### Known issues (documented, not yet fixed)

- `requirements.txt` is a conda win-64 `@EXPLICIT` export, not a pip file; `pip install -r requirements.txt` fails. See [SETUP.md](SETUP.md).

## Upstream baseline

- Last upstream activity: September 2020 (README update `e7e1bdb`, merge `cdd55d3`); the numbered six-stage pipeline, `GeoNetEnv.yml` (Python 3.7.7 win-64), and the `Old_Version/` boundary all date from this era.
- scikit-fmm pin rationale inherited by the fork: the conda env's `pip:` section pins `scikit-fmm==2021.7.8` because 2021.1.20+ fixes the negative-discriminant march abort (scikit-fmm issue #18) that breaks stage 5 on the 2019.1.30 release the original README's build-from-source instructions installed.
