"""Regression tests: per-basin geodesic distance scatter in Fast_Marching.

Upstream (e7e1bdb) scatters each catchment's solved travel time into the
full-raster ``geodesicDistanceArray`` inside the per-basin loop, so every
basin contributes its own distances. The domain-guard refactor (merged as
cbb071f, PR #3) deleted that in-loop scatter together with the
``except ValueError`` block above it, leaving only a dedented assignment
after the loop: every basin was still solved, but only the final basin's
distances survived and all earlier basins came out NaN. On the frozen v6
benchmark inputs this produced 120832/510987/16313035 NaNs at
512^2/1024^2/4096^2 instead of the upstream 0/0/13838.

These tests pin the restored contract directly against ``Fast_Marching``
(not ``main``, so no raster I/O and no GRASS):

- every pixel of every solved basin carries a finite geodesic distance,
- each basin's values equal that basin's own solve (no cross-basin mixing),
- pixels outside every solved basin stay NaN (upstream Inf-to-NaN policy),
- the raster handed to the writer equals the returned array.

Only numpy + skfmm are real; every heavy geospatial dependency is stubbed
(the same pattern as test_fast_marching.py).
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub_modules():
    """Stub GeoNet's heavy geospatial dependencies before import."""
    def plain_module(name):
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module

    # osgeo (gdal, osr, ogr) as used by pygeonet_rasterio
    osgeo = plain_module('osgeo')
    for submodule in ('gdal', 'osr', 'ogr'):
        sub = types.ModuleType('osgeo.' + submodule)
        setattr(osgeo, submodule, sub)
        sys.modules['osgeo.' + submodule] = sub

    # rasterio.crs.CRS as used by pygeonet_rasterio
    rasterio = plain_module('rasterio')
    rasterio_crs = plain_module('rasterio.crs')
    rasterio_crs.CRS = type('CRS', (), {})
    rasterio.crs = rasterio_crs

    # matplotlib.pyplot / matplotlib.cm as used by pygeonet_plot
    # (pygeonet_plot reads cm.coolwarm as a default argument at import time)
    matplotlib = plain_module('matplotlib')
    matplotlib.pyplot = plain_module('matplotlib.pyplot')
    matplotlib.cm = plain_module('matplotlib.cm')
    matplotlib.cm.__getattr__ = lambda name: name
    matplotlib.use = lambda *a, **k: None

    # numba njit/prange as used by pygeonet_fast_marching (the decorated
    # selection kernel is never called here, so a passthrough is enough)
    numba = plain_module('numba')
    numba.njit = lambda *a, **k: (
        (lambda fn: fn) if not (a and callable(a[0])) else a[0])
    numba.prange = range

    # psutil.virtual_memory as used by Fast_Marching
    psutil = plain_module('psutil')
    psutil.virtual_memory = lambda: types.SimpleNamespace(total=0)

    # dask + dask.distributed.Client
    dask = plain_module('dask')
    dask.distributed = plain_module('dask.distributed')
    dask.distributed.Client = type('Client', (), {})

    # pygeonet_prepare parses the project cfg at import time; earlier test
    # modules may have stubbed it first, so only create it when absent.
    if 'pygeonet_prepare' not in sys.modules:
        plain_module('pygeonet_prepare')
    plain_module('pygeonet_defaults')


_install_stub_modules()

import numpy as np  # noqa: E402
import skfmm  # noqa: E402
import pygeonet_fast_marching as fmm  # noqa: E402

# Fast_Marching reads two path attributes from Parameters when writing its
# output raster. Whichever module object the import chain bound (this file
# standalone, or a stub left by an earlier test module), set the attributes
# on the live object fmm actually references.
import atexit  # noqa: E402
import tempfile  # noqa: E402

_Parameters = fmm.Parameters
_Parameters.geonetResultsDir = tempfile.mkdtemp(prefix='geonet-scatter-tests-')
_Parameters.demFileName = 'dem.tif'
atexit.register(lambda: __import__('shutil').rmtree(
    _Parameters.geonetResultsDir, ignore_errors=True))

# Three basins with distinct shapes so their travel-time fields differ;
# start points sit at each rectangle's top-left corner inside its basin.
SHAPE = (32, 48)
RECTS = [(2, 10, 2, 12), (2, 8, 20, 34), (14, 30, 4, 40)]
COST = 2.0  # uniform reciprocal local cost inside every basin


def build_case():
    """Basin raster, start-point list, flow (FAC), and cost field."""
    basin = np.zeros(SHAPE, dtype=np.float64)
    flow = np.zeros(SHAPE, dtype=np.float64)
    for i, (r0, r1, c0, c1) in enumerate(RECTS):
        basin[r0:r1, c0:c1] = i + 1
        flow[r0:r1, c0:c1] = 10.0 * (i + 1)
    splist = np.array([[r0 for r0, r1, c0, c1 in RECTS],
                       [c0 for r0, r1, c0, c1 in RECTS]], dtype=np.int64)
    reciprocal = np.full(SHAPE, COST, dtype=np.float64)
    return basin, splist, flow, reciprocal


def reference_solve(index):
    """The exact phi/speed construction and solve Fast_Marching performs."""
    phi = np.zeros(SHAPE)
    speed = np.zeros(SHAPE)
    mask = np.zeros(SHAPE, dtype=bool)
    r0, r1, c0, c1 = RECTS[index]
    mask[r0:r1, c0:c1] = True
    phi[mask] = 1
    speed[mask] = COST
    phi[splist[0, index], splist[1, index]] = -1
    return skfmm.travel_time(phi, speed, dx=.01)


basin, splist, flow, reciprocal = build_case()


class FastMarchingScatterTests(unittest.TestCase):
    """Every solved basin must land in the returned geodesic distance array."""

    def setUp(self):
        writes = []
        self._original_write = fmm.write_geotif_generic
        fmm.write_geotif_generic = lambda *args: writes.append(args)
        self.addCleanup(setattr, fmm, 'write_geotif_generic',
                        self._original_write)
        self.writes = writes
        self.result = fmm.Fast_Marching(splist, basin, flow, reciprocal)

    def test_every_solved_basin_has_finite_distances(self):
        """The cbb071f regression left all-but-last basins NaN."""
        for i, (r0, r1, c0, c1) in enumerate(RECTS):
            field = self.result[r0:r1, c0:c1]
            self.assertTrue(
                np.all(np.isfinite(field)),
                'basin {0} carries non-finite geodesic distances'.format(i + 1))

    def test_pixels_outside_every_basin_stay_nan(self):
        """Unsolved pixels are Inf then converted to NaN, upstream policy."""
        outside = self.result[basin == 0]
        self.assertTrue(np.all(np.isnan(outside)))
        self.assertFalse(np.isinf(self.result).any())

    def test_each_basin_carries_its_own_solve(self):
        """No cross-basin mixing: values match a per-basin reference solve."""
        for i in range(len(RECTS)):
            mask = basin == i + 1
            reference = reference_solve(i)
            self.assertTrue(
                np.array_equal(self.result[mask], reference[mask]),
                'basin {0} does not match its own travel-time solve'.format(
                    i + 1))

    def test_written_raster_equals_returned_array(self):
        """The raster handed to the writer is the returned array."""
        self.assertEqual(len(self.writes), 1)
        written_array, outfilepath, outfilename = self.writes[0]
        self.assertTrue(
            np.array_equal(written_array, self.result, equal_nan=True))
        self.assertEqual(outfilename, 'dem_geodesicDistance.tif')


if __name__ == '__main__':
    unittest.main()
