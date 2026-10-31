# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate the seeded synthetic .npy goldens consumed by the Julia suite.

Runs the ORIGINAL pygeonet kernels (pygeonet_nonlinear_filter,
pygeonet_slope_curvature, pygeonet_fast_marching) on seeded synthetic
inputs and dumps every stage input/output to julia/test/goldens/ together
with a manifest recording the exact reference stack, seeds, kernel
arguments, and per-file SHA-256 hashes.

Import-time-only dependencies that are absent here (osgeo GDAL bindings,
rasterio.crs, statsmodels, skfmm, dask) are stubbed exactly the way
tests/test_stage5_determinism.py stubs osgeo: none of the kernels under
test touch them at call time. matplotlib, numba, and psutil are real.

Usage:  python3 julia/tools/generate_goldens.py
"""

import configparser
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import scipy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "test", "goldens")

GOLDEN_SEED = 20261030
PIXEL_SCALE = 30.0            # float32-exact; Parameters.demPixelScale for the fixtures
PM_KERNEL_WIDTH = 5           # gaussian kernel width for the smoothing golden
PM_SIGMA_SQUARED = 0.05       # pygeonet_defaults.diffusionSigmaSquared
PM_NITER = 50                 # pygeonet_defaults.nFilterIterations
PM_GAMMA = 0.1                # pygeonet_defaults.diffusionTimeIncrement
BASIN_ELEMENTS = 2            # pygeonet_prepare.numBasinsElements
THRESHOLD = 0.1               # pygeonet_defaults.thresholdPercentAreaForDelineation


# --------------------------------------------------------------------------
# Import stubs (only what the pygeonet import chain needs; never used at call
# time by the kernels under test).
# --------------------------------------------------------------------------

def _install_stubs():
    try:
        import osgeo.gdal  # noqa: F401
    except ImportError:
        osgeo = types.ModuleType("osgeo")
        for sub in ("gdal", "osr", "ogr"):
            mod = types.ModuleType("osgeo.%s" % sub)
            sys.modules["osgeo.%s" % sub] = mod
            setattr(osgeo, sub, mod)
        sys.modules["osgeo"] = osgeo

    try:
        import rasterio.crs  # noqa: F401
    except ImportError:
        rasterio = types.ModuleType("rasterio")
        crs_mod = types.ModuleType("rasterio.crs")
        crs_mod.CRS = object
        sys.modules["rasterio.crs"] = crs_mod
        rasterio.crs = crs_mod
        sys.modules["rasterio"] = rasterio

    try:
        import statsmodels.api  # noqa: F401
    except ImportError:
        statsmodels = types.ModuleType("statsmodels")
        api = types.ModuleType("statsmodels.api")
        api.ProbPlot = object
        sys.modules["statsmodels.api"] = api
        statsmodels.api = api
        sys.modules["statsmodels"] = statsmodels

    try:
        import skfmm  # noqa: F401
    except ImportError:
        skfmm = types.ModuleType("skfmm")
        skfmm.travel_time = None
        sys.modules["skfmm"] = skfmm

    try:
        import dask.distributed  # noqa: F401
    except ImportError:
        dask = types.ModuleType("dask")
        dist = types.ModuleType("dask.distributed")
        dist.Client = object
        sys.modules["dask.distributed"] = dist
        dask.distributed = dist
        sys.modules["dask"] = dask


def _ensure_import_env():
    """Create the scratch GeoNet cfg pygeonet_prepare parses at import time."""
    if glob.glob(os.path.join(REPO_ROOT, "GeoNet*.cfg")):
        return None, None, None
    home = tempfile.mkdtemp(prefix="geonet-julia-goldens-")
    cfg_path = os.path.join(REPO_ROOT, "GeoNet_julia_goldens.cfg")
    cfg = configparser.ConfigParser()
    cfg["Section"] = {
        "geofloodhomedir": home,
        "projectname": "juliagoldens",
        "dem_name": "dem",
        "Input_dir": "GeoInputs",
        "Output_dir": "GeoOutputs",
    }
    with open(cfg_path, "w") as fh:
        cfg.write(fh)
    return cfg_path, home, None


def _cleanup(cfg_path, home):
    if cfg_path and os.path.exists(cfg_path):
        os.remove(cfg_path)
    if home and os.path.isdir(home):
        shutil.rmtree(home, ignore_errors=True)


# --------------------------------------------------------------------------
# Golden writing helpers
# --------------------------------------------------------------------------

def save(name, array):
    path = os.path.join(GOLDEN_DIR, name)
    np.save(path, array)
    return name


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Fixture builders (build_case mirrors tests/test_stage5_determinism.py)
# --------------------------------------------------------------------------

def build_case(ny, nx, rects, dtype=np.float64):
    """Basin raster from explicit (r0, r1, c0, c1) rectangles, one outlet each."""
    basin = np.zeros((ny, nx), dtype=dtype)
    outlets = []
    for i, (r0, r1, c0, c1) in enumerate(rects):
        basin[r0:r1, c0:c1] = i + 1
        outlets.append((r0, c0))
    splist = np.transpose(np.array(outlets, dtype=np.int64)).reshape(2, -1)
    return basin, splist


def make_dem():
    """Seeded synthetic DEM with NaN holes, a below-flag patch, and a -9998 flag patch."""
    rng = np.random.default_rng(GOLDEN_SEED)
    ny, nx = 96, 96
    yy, xx = np.mgrid[0:ny, 0:nx]
    dem = (60.0
           + 12.0 * np.sin(xx / 7.3 + 1.1) * np.cos(yy / 9.7)
           + 0.35 * xx + 0.21 * yy
           + rng.normal(0.0, 0.8, size=(ny, nx))).astype(np.float32)
    holes = rng.choice(ny * nx, size=170, replace=False)
    flat = dem.ravel()
    flat[holes] = np.nan
    dem = flat.reshape(ny, nx)
    # The -9998 flag patch itself is NOT NaN (only values strictly below the
    # flag convert in main(): nanDemArray[nanDemArray < demNanFlag] = np.nan).
    dem[3:7, 80:86] = np.float32(-9998.0)
    dem[50:53, 10:13] = np.float32(-9999.5)   # below-flag cells -> NaN below
    dem[dem < -9998.0] = np.nan
    return dem


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    cfg_path, home, _ = _ensure_import_env()
    try:
        _install_stubs()
        import pygeonet_prepare as Parameters  # noqa: F401  (cfg parsed at import)
        import pygeonet_nonlinear_filter as nf
        import pygeonet_slope_curvature as sc
        import pygeonet_fast_marching as fm

        Parameters.demPixelScale = PIXEL_SCALE

        dem = make_dem()

        # ---- stage 1: Perona-Malik chain --------------------------------
        smoothed = nf.simple_gaussian_smoothing(dem.copy(), PM_KERNEL_WIDTH,
                                                PM_SIGMA_SQUARED)
        edge_threshold = nf.lambda_nonlinear_filter(dem.copy())
        filtered = nf.anisodiff(dem.copy(), PM_NITER, edge_threshold, PM_GAMMA,
                                (PIXEL_SCALE, PIXEL_SCALE), 2)
        filtered_opt1 = nf.anisodiff(dem.copy(), PM_NITER, edge_threshold,
                                     PM_GAMMA, (PIXEL_SCALE, PIXEL_SCALE), 1)

        # ---- stage 2: slope and curvature on the filtered DEM -----------
        slope = sc.compute_dem_slope(filtered.copy(), PIXEL_SCALE)
        curvature, curv_mean, curv_std = sc.compute_dem_curvature(
            filtered.copy(), PIXEL_SCALE, "geometric")
        curvature_lap, _, _ = sc.compute_dem_curvature(
            filtered.copy(), PIXEL_SCALE, "laplacian")

        # ---- stage 5: start-point kernel (the three oracle geometries) --
        grid_rects = [(r, r + 16, c, c + 16)
                      for r in range(0, 96, 16) for c in range(0, 96, 16)]
        cases = {
            "grid36": build_case(96, 96, grid_rects),
            "mixed20": build_case(20, 20, [
                (0, 3, 0, 3), (0, 3, 3, 6), (0, 3, 6, 9),
                (3, 4, 0, 3), (3, 4, 3, 6),
                (4, 5, 0, 2), (4, 5, 3, 5), (5, 6, 0, 1), (5, 6, 1, 2),
            ]),
            "edge1x32": build_case(1, 32, [(0, 1, 0, 3)]),
        }

        # ---- mquantiles reference micro-fixture -------------------------
        # Pins the scipy mquantiles (alphap=betap=0.4) semantics the Julia
        # port of lambda_nonlinear_filter must reproduce.
        rng2 = np.random.default_rng(GOLDEN_SEED + 1)
        sizes = list(range(1, 65)) + [100, 500, 1000]
        vals, quantiles, offsets, running = [], [], [0], 0
        for n in sizes:
            v = rng2.normal(0.0, 1.0, size=n).astype(np.float32)
            from scipy.stats.mstats import mquantiles
            q = mquantiles(np.absolute(v), 0.9).item()
            vals.append(v)
            quantiles.append(q)
            running += n
            offsets.append(running)

        # ---- save everything --------------------------------------------
        save("pm_input_dem.npy", dem)
        save("pm_gaussian_smoothing.npy", smoothed)
        save("pm_lambda.npy", np.array([edge_threshold], dtype=np.float64))
        save("pm_filtered.npy", filtered)
        save("pm_filtered_opt1.npy", filtered_opt1)
        save("slope_output.npy", slope)
        save("curvature_output.npy", curvature)
        save("curvature_laplacian.npy", curvature_lap)
        save("curvature_stats.npy",
             np.array([curv_mean, curv_std], dtype=np.float64))
        for case, (basin, splist) in cases.items():
            iter_total = np.arange(0, splist.shape[1]).size
            keep = fm.Fast_Marching_Start_Point_Identification(
                None, basin, splist, float(basin.size), BASIN_ELEMENTS,
                THRESHOLD, iter_total)
            fmm = fm.fmm_list_creation(splist, keep)
            save("sp_%s_basin.npy" % case, basin)
            save("sp_%s_splist.npy" % case, splist)
            save("sp_%s_keep.npy" % case, keep.astype(np.uint8))
            save("sp_%s_fmm.npy" % case, fmm)
        save("mq_vals.npy", np.concatenate(vals))
        save("mq_offsets.npy", np.array(offsets, dtype=np.int64))
        save("mq_sizes.npy", np.array(sizes, dtype=np.int64))
        save("mq_quantiles.npy", np.array(quantiles, dtype=np.float64))

        manifest = {
            "license": "GPL-3.0-or-later",
            "seed": GOLDEN_SEED,
            "pixel_scale": PIXEL_SCALE,
            "pm_args": {
                "kernel_width": PM_KERNEL_WIDTH,
                "sigma_squared": PM_SIGMA_SQUARED,
                "niter": PM_NITER,
                "gamma": PM_GAMMA,
                "step": [PIXEL_SCALE, PIXEL_SCALE],
                "option_default": 2,
            },
            "start_point_args": {
                "basin_elements": BASIN_ELEMENTS,
                "threshold_percent_area": THRESHOLD,
            },
            "reference_stack": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "numba": sys.modules["numba"].__version__,
            },
            "source_commit": os.popen(
                "git -C %s rev-parse HEAD" % REPO_ROOT).read().strip(),
            "notes": [
                "pm_input_dem.npy is the post-flag-conversion NaN DEM the kernels receive",
                "keep masks are stored as uint8 (0/1)",
                "lambda and curvature_stats are stored as float64",
                "mq_* pins scipy mquantiles defaults (alphap=betap=0.4) on |values| at p=0.9",
            ],
            "files": {},
        }
        for name in sorted(os.listdir(GOLDEN_DIR)):
            if name.endswith(".npy"):
                manifest["files"][name] = sha256_file(os.path.join(GOLDEN_DIR, name))
        with open(os.path.join(GOLDEN_DIR, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")

        print("goldens written to %s (%d .npy files)"
              % (GOLDEN_DIR, len(manifest["files"])))
        print("edge threshold (lambda) = %.17g" % edge_threshold)
        print("curvature mean/std = %.17g / %.17g" % (curv_mean, curv_std))
    finally:
        _cleanup(cfg_path, home)


if __name__ == "__main__":
    main()
