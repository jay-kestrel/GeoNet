"""Regression tests: stage-5 Numba kernel determinism (GeoNet todo_vj7M7zN1).

``Fast_Marching_Start_Point_Identification`` historically appended to two shared
Python lists inside ``prange``: outputs came out in thread-schedule order (not
label order), appends were lost, and the heap corrupted (SIGSEGV / SIGABRT —
reproduced 26/26 fresh parallel processes on one geometry sweep).

The kernel now performs disjoint per-label writes into a boolean mask, and
``fmm_list_creation`` gathers start points in label order (the deterministic
serial reduction). These tests pin that contract:

- repeated parallel runs are byte-identical,
- outputs match a pure-numpy label-order reference exactly,
- both selection gates (percent of DEM area, minimum basin elements) are
  exercised, including cases where each gate alone excludes a basin,
- degenerate array-edge shapes (1-row, 1-column, 1-pixel, zero-outlet) run.

Plain ``unittest`` so both ``python -m unittest discover -s tests`` (the CI
command) and pytest run them. Tests run under the default numba thread count
on purpose — the parallel path is what must be deterministic. Do not set
NUMBA_NUM_THREADS=1 here to make them pass.
"""
import atexit
import configparser
import glob
import os
import shutil
import sys
import tempfile
import types
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# pygeonet's modules read a project cfg at import time (pygeonet_prepare globs
# GeoNet*.cfg in the repo root) and import GDAL bindings from ``osgeo``. Set up
# both before importing the kernel under test: create a scratch cfg when none
# exists, and fall back to a minimal osgeo stub when the GDAL bindings are not
# installed (the stage-5 numba kernel does no raster I/O).
_CFG_PATH = None
_CFG_HOME = None


def _cleanup_cfg():
    if _CFG_PATH and os.path.exists(_CFG_PATH):
        os.remove(_CFG_PATH)
    if _CFG_HOME and os.path.isdir(_CFG_HOME):
        shutil.rmtree(_CFG_HOME, ignore_errors=True)


def _ensure_import_env():
    global _CFG_PATH, _CFG_HOME
    try:
        import osgeo.gdal  # noqa: F401
    except ImportError:
        _osgeo = types.ModuleType("osgeo")
        for _sub in ("gdal", "osr", "ogr"):
            _mod = types.ModuleType("osgeo.%s" % _sub)
            sys.modules["osgeo.%s" % _sub] = _mod
            setattr(_osgeo, _sub, _mod)
        sys.modules["osgeo"] = _osgeo
    if glob.glob(os.path.join(REPO_ROOT, "GeoNet*.cfg")):
        return  # an existing user/project cfg wins; do not touch it
    _CFG_HOME = tempfile.mkdtemp(prefix="geonet-stage5-tests-")
    _CFG_PATH = os.path.join(REPO_ROOT, "GeoNet_stage5tests.cfg")
    cfg = configparser.ConfigParser()
    cfg["Section"] = {
        "geofloodhomedir": _CFG_HOME,
        "projectname": "stage5tests",
        "dem_name": "dem",
        "Input_dir": "GeoInputs",
        "Output_dir": "GeoOutputs",
    }
    with open(_CFG_PATH, "w") as fh:
        cfg.write(fh)
    atexit.register(_cleanup_cfg)


_ensure_import_env()

try:
    import numpy as np
    import pygeonet_fast_marching as fm
except ImportError as exc:  # env without the numba stack: skip, do not fail
    raise unittest.SkipTest(
        "numba stack unavailable (%s); stage-5 determinism tests skipped" % exc)

BASIN_ELEMENTS = 2   # Parameters.numBasinsElements
THRESHOLD = 0.1      # defaults.thresholdPercentAreaForDelineation


def build_case(ny, nx, rects, dtype=np.float64):
    """Basin raster from explicit (r0, r1, c0, c1) rectangles, one outlet each.

    Basin ids are 1..B in the order given (matching GRASS basin numbering and
    main()'s argwhere-derived outlet list). Remaining pixels stay 0.
    """
    basin = np.zeros((ny, nx), dtype=dtype)
    outlets = []
    for i, (r0, r1, c0, c1) in enumerate(rects):
        basin[r0:r1, c0:c1] = i + 1
        outlets.append((r0, c0))
    splist = np.transpose(np.array(outlets, dtype=np.int64)).reshape(2, -1)
    return basin, splist


def reference_keep(basin, splist, basin_elements=BASIN_ELEMENTS,
                   threshold=THRESHOLD):
    """Pure-numpy ground truth: per-label count, label-order keep mask."""
    iter_total = splist.shape[1]
    keep = np.zeros(iter_total, dtype=bool)
    nDempixels = float(basin.size)
    for label in range(iter_total):
        n = np.count_nonzero(basin == label + 1)
        if (n * 100.00001 / nDempixels > threshold) and (n > basin_elements):
            keep[label] = True
    return keep


def run_kernel(basin, splist):
    iter_total = np.arange(0, splist.shape[1]).size
    return fm.Fast_Marching_Start_Point_Identification(
        None, basin, splist, float(basin.size), BASIN_ELEMENTS, THRESHOLD,
        iter_total)


class DeterminismTests(unittest.TestCase):
    """Repeated parallel runs must be byte-identical and match the reference."""

    @classmethod
    def setUpClass(cls):
        # 36 basins of 16x16 on a 96x96 DEM — every basin qualifies.
        rects = [(r, r + 16, c, c + 16)
                 for r in range(0, 96, 16) for c in range(0, 96, 16)]
        cls.basin, cls.splist = build_case(96, 96, rects)

    def test_repeated_runs_identical(self):
        first = run_kernel(self.basin, self.splist)
        first_fmm = fm.fmm_list_creation(self.splist, first)
        for _ in range(12):
            keep = run_kernel(self.basin, self.splist)
            fmm = fm.fmm_list_creation(self.splist, keep)
            self.assertTrue(np.array_equal(keep, first))
            self.assertEqual(fmm.tobytes(), first_fmm.tobytes())
            self.assertEqual(fmm.shape, first_fmm.shape)

    def test_matches_pure_numpy_reference(self):
        keep = run_kernel(self.basin, self.splist)
        ref = reference_keep(self.basin, self.splist)
        self.assertEqual(keep.dtype, np.bool_)
        self.assertTrue(np.array_equal(keep, ref))
        self.assertTrue(keep.all())  # every 16x16 basin is 2.78% of the DEM
        fmm = fm.fmm_list_creation(self.splist, keep)
        self.assertTrue(np.array_equal(fmm, self.splist[:, ref]))


class SelectionGateTests(unittest.TestCase):
    """Isolate each exclusion gate on a 20x20 DEM.

    Percent gate passes for any n >= 1 (0.25% minimum > 0.1%), so the 1-px and
    2-px basins below are excluded by the minimum-elements gate alone, while
    the 3-px strips qualify (0.75%, n=3 > 2). Basins 6-9 are all excluded.
    """

    def test_mixed_qualification_gates(self):
        rects = [
            (0, 3, 0, 3),        # 1: 9 px  -> qualify
            (0, 3, 3, 6),        # 2: 9 px  -> qualify
            (0, 3, 6, 9),        # 3: 9 px  -> qualify
            (3, 4, 0, 3),        # 4: 3 px strip -> qualify
            (3, 4, 3, 6),        # 5: 3 px strip -> qualify
            (4, 5, 0, 2),        # 6: 2 px  -> elements gate only
            (4, 5, 3, 5),        # 7: 2 px  -> elements gate only
            (5, 6, 0, 1),        # 8: 1 px  -> elements gate only
            (5, 6, 1, 2),        # 9: 1 px  -> elements gate only
        ]
        basin, splist = build_case(20, 20, rects)
        keep = run_kernel(basin, splist)
        self.assertTrue(np.array_equal(keep, reference_keep(basin, splist)))
        expected = np.zeros(splist.shape[1], dtype=bool)
        expected[:5] = True
        self.assertEqual(keep.shape, (splist.shape[1],))
        self.assertTrue(np.array_equal(keep, expected))
        fmm = fm.fmm_list_creation(splist, keep)
        self.assertEqual(fmm.shape, (2, 5))
        # label order: the five qualifying outlets, rows then cols
        self.assertTrue(np.array_equal(fmm[0], [0, 0, 0, 3, 3]))
        self.assertTrue(np.array_equal(fmm[1], [0, 3, 6, 0, 3]))


class EdgeShapeTests(unittest.TestCase):
    """Degenerate raster edges: 1-row, 1-column, 2x2, 1-pixel, zero outlets."""

    def test_array_edge_shapes(self):
        for ny, nx in [(1, 32), (32, 1), (2, 2), (1, 1)]:
            with self.subTest(shape=(ny, nx)):
                rects = [(0, 1, 0, 3)] if nx >= 3 else [(0, 1, 0, 1)]
                basin, splist = build_case(ny, nx, rects)
                keep = run_kernel(basin, splist)
                self.assertTrue(np.array_equal(keep, reference_keep(basin, splist)))
                fmm = fm.fmm_list_creation(splist, keep)
                self.assertEqual(fmm.shape[0], 2)
                self.assertEqual(fmm.shape[1], int(keep.sum()))

    def test_zero_outlet_list(self):
        basin = np.zeros((8, 8), dtype=np.float64)
        splist = np.zeros((2, 0), dtype=np.int64)
        keep = run_kernel(basin, splist)
        self.assertEqual(keep.shape, (0,))
        fmm = fm.fmm_list_creation(splist, keep)
        self.assertEqual(fmm.shape, (2, 0))


class BasinDtypeTests(unittest.TestCase):
    """Basin rasters arrive as Int16 GeoTIFFs (or float64 after read) — the
    kernel must treat all identically."""

    def test_basin_raster_dtypes(self):
        rects = [(0, 8, 0, 8), (0, 8, 8, 16), (8, 16, 0, 8), (8, 16, 8, 16)]
        for dtype in (np.float64, np.int16, np.int32):
            with self.subTest(dtype=np.dtype(dtype).name):
                basin, splist = build_case(16, 16, rects, dtype=dtype)
                keep = run_kernel(basin, splist)
                self.assertTrue(np.array_equal(keep, reference_keep(basin, splist)))
                self.assertTrue(keep.all())


if __name__ == "__main__":
    unittest.main()
