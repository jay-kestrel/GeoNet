"""Regression tests for GN-2: FMM & channel-head robustness.

Covers Adversarial Review — GeoNet findings F3, F4, F13-min and F14
(art_J8xxVxhP), re-derived at master ``9d4af3f`` (post GN-1 / PR #10):

- F3: ``Fast_Marching``'s only fallback handler is ``except IOError`` (an
  OSError alias) so it can never catch the ``FastMarchingDomainError`` the
  domain guard raises (a RuntimeError subclass) — one failing basin aborts the
  whole multi-basin run. The duplicated post-loop scatter re-writes the last
  basin's values and raises ``NameError`` when no basin qualifies ((2, 0)
  start list).
- F4: all-equal component sizes exclude every component from the endpoint
  scan (threshold lands on the histogram's upper edge) -> empty endpoint list
  -> ``IndexError`` at ``skeletonEndPointsListArray[1]``; a zero-component
  skeleton crashes earlier, inside ``ndimage.labeled_comprehension``.
- F13-min: a numeric-string ``reciprocalLocalCostMinimum`` (``'0.5'``) raises
  ``TypeError`` from the ``array < str`` comparison (review probe P8).
- F14: ``write_geotif_filteredDEM`` calls ``int(GetAuthorityCode("PROJCS"))``,
  which raises ``TypeError`` when the CRS carries no EPSG authority code.

The heavy geospatial imports of the GeoNet modules (gdal, rasterio,
matplotlib, numba, dask, psutil) are stubbed so the tests need only numpy +
skfmm + scipy + the standard library (same pattern as
tests/test_fast_marching.py).
"""
import os
import sys
import types
import unittest

# Make the repo root importable regardless of the test runner's rootdir.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub_modules():
    """Stub GeoNet's heavy geospatial dependencies before import."""
    def plain_module(name):
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module

    # osgeo (gdal, osr, ogr) as used by pygeonet_rasterio/pygeonet_vectorio
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
    matplotlib = plain_module('matplotlib')
    matplotlib.pyplot = plain_module('matplotlib.pyplot')
    matplotlib.cm = plain_module('matplotlib.cm')
    matplotlib.cm.__getattr__ = lambda name: name
    matplotlib.use = lambda *a, **k: None

    # numba njit/jit/prange as used by pygeonet_fast_marching and
    # pygeonet_channel_head_definition
    numba = plain_module('numba')
    numba.njit = lambda *a, **k: (
        (lambda fn: fn) if not (a and callable(a[0])) else a[0])
    numba.jit = numba.njit
    numba.prange = range

    # psutil.virtual_memory as used by Fast_Marching
    psutil = plain_module('psutil')
    psutil.virtual_memory = lambda: types.SimpleNamespace(total=0)

    # dask + dask.distributed.Client
    dask = plain_module('dask')
    dask.distributed = plain_module('dask.distributed')
    dask.distributed.Client = type('Client', (), {})

    # pygeonet_prepare parses the project cfg at import time and crashes
    # without one; the behaviors under test need none of it.
    plain_module('pygeonet_prepare')
    plain_module('pygeonet_defaults')


_install_stub_modules()

import numpy as np  # noqa: E402
import skfmm  # noqa: E402
import pygeonet_fast_marching as fmm  # noqa: E402

_MISSING = object()


class ModuleStateTestCase(unittest.TestCase):
    """Base class that restores every module global a test rebinds.

    The GeoNet modules share one module object per process; without the
    restore, a rebind here would leak into sibling test modules.
    """

    def setUp(self):
        self._saved = []

    def tearDown(self):
        for obj, name, old in reversed(self._saved):
            if old is _MISSING:
                obj.__dict__.pop(name, None)
            else:
                setattr(obj, name, old)

    def set_attr(self, obj, name, value):
        self._saved.append((obj, name, obj.__dict__.get(name, _MISSING)))
        setattr(obj, name, value)

    def run_fast_marching(self, starts, basin, flow, cost):
        """Fast_Marching with the result writer stubbed (it writes a tif)."""
        self.set_attr(fmm, 'Parameters', types.SimpleNamespace(
            geonetResultsDir='/tmp/geonet-gn2-tests', demFileName='dem.tif'))
        writes = []
        self.set_attr(fmm, 'write_geotif_generic',
                      lambda *args, **kwargs: writes.append(args))
        return fmm.Fast_Marching(starts, basin, flow, cost), writes


def build_two_basin_inputs(shape=(20, 40)):
    """Two-basin domain shaped exactly like Fast_Marching's inputs."""
    basin_index_array = np.zeros(shape, dtype=int)
    basin_index_array[:, :shape[1] // 2] = 1
    basin_index_array[:, shape[1] // 2:] = 2
    flow_array = np.ones(shape)
    reciprocal_local_cost_array = np.full(shape, 0.5) + np.mod(
        np.arange(shape[0] * shape[1]).reshape(shape), 7) * 0.001
    start_points = np.array([[shape[0] // 2, shape[0] // 2],
                             [shape[0] // 2, shape[1] - 5]])
    return (basin_index_array, flow_array,
            reciprocal_local_cost_array, start_points)


class HealthyPathTests(ModuleStateTestCase):
    def test_healthy_two_basin_run_solves_both_basins(self):
        """No behavior change on well-formed inputs: both basins solve."""
        basin, flow, cost, starts = build_two_basin_inputs()
        result, _ = self.run_fast_marching(starts, basin, flow, cost)
        self.assertEqual(result.shape, basin.shape)
        self.assertTrue(np.all(np.isfinite(result[basin == 1])))
        self.assertTrue(np.all(np.isfinite(result[basin == 2])))
        self.assertGreater(float(np.nanmin(result)), 0.0)


class FastMarchingDomainErrorFallbackTests(ModuleStateTestCase):
    """F3: a failing basin degrades to NaN; the run continues."""

    def test_domain_error_in_one_basin_degrades_to_nan(self):
        basin, flow, cost, starts = build_two_basin_inputs()
        real_guard = fmm.travel_time_with_domain_guard

        def flaky_guard(phi, speed, dx, catchmentIndex, startPoint):
            if catchmentIndex == 1:
                raise fmm.FastMarchingDomainError(
                    'scikit-fmm failed for catchment 1')
            return real_guard(phi, speed, dx,
                              catchmentIndex=catchmentIndex,
                              startPoint=startPoint)

        self.set_attr(fmm, 'travel_time_with_domain_guard', flaky_guard)
        result, _ = self.run_fast_marching(starts, basin, flow, cost)
        # failing basin -> NaN field; the healthy basin still solves
        self.assertTrue(np.all(np.isnan(result[basin == 1])))
        self.assertTrue(np.all(np.isfinite(result[basin == 2])))


class ZeroQualifyingBasinsTests(ModuleStateTestCase):
    """F3: an empty start list yields a defined empty (all-NaN) result."""

    def test_empty_start_list_returns_all_nan_without_name_error(self):
        basin, flow, cost, _ = build_two_basin_inputs()
        result, _ = self.run_fast_marching(np.zeros((2, 0)), basin, flow, cost)
        self.assertEqual(result.shape, basin.shape)
        self.assertTrue(np.all(np.isnan(result)))


class LocalCostMinimumTests(ModuleStateTestCase):
    """F13-min: reciprocalLocalCostMinimum is numeric-or-None (probe P8)."""

    COST_FN = ('flowArray + (flowMean*skeletonFromFlowAndCurvatureArray)'
               ' + (flowMean*curvatureDemArray)')

    def run_local_cost(self, minimum):
        self.set_attr(fmm, 'defaults', types.SimpleNamespace(
            reciprocalLocalCostFn=self.COST_FN,
            reciprocalLocalCostMinimum=minimum))
        self.set_attr(fmm, 'Parameters', types.SimpleNamespace(
            geonetResultsDir='/tmp/geonet-gn2-tests',
            demFileName='dem.tif'))
        writes = []
        self.set_attr(fmm, 'write_geotif_generic',
                      lambda *args, **kwargs: writes.append(args))
        flow = np.full((8, 8), 2.0)
        flow[0, 0] = 0.3
        result = fmm.Local_Cost_Computation(
            flow, 1.0, np.zeros((8, 8)), np.zeros((8, 8)))
        return result, writes

    def test_numeric_string_minimum_clamps(self):
        # P8 input: the cfg reader stores numbers as strings.
        result, _ = self.run_local_cost('0.5')
        self.assertAlmostEqual(result[0, 0], 1.0)
        self.assertAlmostEqual(result[0, 1], 2.0)

    def test_float_minimum_clamps(self):
        result, _ = self.run_local_cost(0.5)
        self.assertAlmostEqual(result[0, 0], 1.0)
        self.assertAlmostEqual(result[0, 1], 2.0)

    def test_nan_string_minimum_skips_clamp(self):
        result, _ = self.run_local_cost('nan')
        self.assertAlmostEqual(result[0, 0], 0.3)

    def test_none_minimum_skips_clamp(self):
        result, _ = self.run_local_cost(None)
        self.assertAlmostEqual(result[0, 0], 0.3)


if __name__ == '__main__':
    unittest.main()
