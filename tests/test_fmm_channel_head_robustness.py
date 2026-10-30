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
import pygeonet_channel_head_definition as ch  # noqa: E402
import pygeonet_rasterio as rio  # noqa: E402

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


def build_equal_size_skeleton(shape=(12, 12)):
    """Four well-separated components of exactly 5 pixels each.

    bins='auto' on the all-equal sizes [5, 5, 5, 5] gives edges
    [4.5, 5.5], so the threshold (5.5) excludes every component -> zero
    channel heads (the F4 crash at current HEAD).
    """
    skel = np.zeros(shape)
    for coords in ([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
                   [(0, 8), (0, 9), (0, 10), (0, 11), (1, 11)],
                   [(5, 0), (6, 0), (7, 0), (8, 0), (9, 0)],
                   [(5, 8), (5, 9), (6, 9), (7, 9), (8, 9)]):
        for (i, j) in coords:
            skel[i, j] = 1
    return skel


def build_mixed_size_skeleton(shape=(12, 12)):
    """One 40-pixel block plus three 3-pixel triominoes (healthy path).

    bins='auto' on sizes [3, 3, 3, 40] gives threshold 21.5: only the
    block qualifies. Its geodesic distances increase as i+j, so exactly
    one pixel — (row 9, col 7) — has no same-label neighbor with a
    strictly greater distance: the single channel head.
    """
    skel = np.zeros(shape)
    block = [(i, j) for i in range(2, 10) for j in range(3, 8)]
    for coords in (block,
                   [(0, 0), (0, 1), (1, 0)],
                   [(0, 9), (0, 10), (1, 9)],
                   [(11, 0), (11, 1), (10, 0)]):
        for (i, j) in coords:
            skel[i, j] = 1
    return skel


def build_mixed_size_geodesic(shape=(12, 12)):
    geodesic = np.full(shape, np.nan)
    geodesic[2:10, 3:8] = np.add.outer(np.arange(2, 10), np.arange(3, 8))
    return geodesic


class ChannelHeadDegenerateTests(ModuleStateTestCase):
    """F4: degenerate skeletons produce defined empty outputs."""

    def setUp(self):
        super().setUp()
        self.shapefile_calls = []
        self.raster_calls = []
        self.set_attr(ch, 'defaults', types.SimpleNamespace(
            doPlot=0, endPointSearchBoxSize=30))
        self.set_attr(ch, 'Parameters', types.SimpleNamespace(
            geonetResultsDir='/tmp/geonet-gn2-tests', demFileName='dem.tif',
            pointFileName='points', pointshapefileName='channelHeads'))
        self.set_attr(ch, 'write_drainage_nodes',
                      lambda *args: self.shapefile_calls.append(args))
        self.set_attr(ch, 'write_geotif_generic',
                      lambda *args: self.raster_calls.append(args))

    def test_all_equal_component_sizes_yield_empty_outputs(self):
        skel = build_equal_size_skeleton()
        xx, yy = ch.Channel_Head_Definition(
            skel, np.full(skel.shape, np.nan))
        self.assertEqual(xx.size, 0)
        self.assertEqual(yy.size, 0)
        # the empty shapefile and zero raster are still written
        self.assertEqual(len(self.shapefile_calls), 1)
        self.assertEqual(len(self.shapefile_calls[0][0]), 0)
        self.assertEqual(len(self.shapefile_calls[0][1]), 0)
        self.assertEqual(len(self.raster_calls), 1)
        np.testing.assert_array_equal(self.raster_calls[0][0],
                                      np.zeros(skel.shape))

    def test_empty_skeleton_yields_empty_outputs(self):
        skel = np.zeros((10, 10))
        xx, yy = ch.Channel_Head_Definition(
            skel, np.full(skel.shape, np.nan))
        self.assertEqual(xx.size, 0)
        self.assertEqual(yy.size, 0)
        self.assertEqual(len(self.shapefile_calls), 1)
        self.assertEqual(len(self.raster_calls), 1)
        np.testing.assert_array_equal(self.raster_calls[0][0],
                                      np.zeros(skel.shape))


class ChannelHeadHealthyPathTests(ModuleStateTestCase):
    """No behavior change on well-formed inputs: one head on the block."""

    def test_mixed_size_skeleton_finds_exactly_the_block_head(self):
        skel = build_mixed_size_skeleton()
        shapefile_calls = []
        raster_calls = []
        self.set_attr(ch, 'defaults', types.SimpleNamespace(
            doPlot=0, endPointSearchBoxSize=30))
        self.set_attr(ch, 'Parameters', types.SimpleNamespace(
            geonetResultsDir='/tmp/geonet-gn2-tests', demFileName='dem.tif',
            pointFileName='points', pointshapefileName='channelHeads'))
        self.set_attr(ch, 'write_drainage_nodes',
                      lambda *args: shapefile_calls.append(args))
        self.set_attr(ch, 'write_geotif_generic',
                      lambda *args: raster_calls.append(args))
        xx, yy = ch.Channel_Head_Definition(
            skel, build_mixed_size_geodesic())
        self.assertEqual(list(xx), [7])
        self.assertEqual(list(yy), [9])
        self.assertEqual(len(shapefile_calls), 1)
        expected_raster = np.zeros(skel.shape)
        expected_raster[9, 7] = 1
        np.testing.assert_array_equal(raster_calls[0][0], expected_raster)


class SkeletonSizeThresholdTests(unittest.TestCase):
    """F4: the threshold helper's selection and degenerate fallback."""

    def test_healthy_selection_picks_first_edge_above_quantile(self):
        edges = np.array([12.0, 31.0, 50.0, 69.0, 88.0])
        self.assertEqual(ch.select_skeleton_size_threshold(edges), 50.0)

    def test_degenerate_equal_edges_fall_back_to_defined_threshold(self):
        # review probe P3: the strict selection over equal edges is empty;
        # the old inline code raised IndexError at edges[test_thresh[0]].
        edges = np.array([4.0, 4.0])
        threshold = ch.select_skeleton_size_threshold(edges)
        self.assertTrue(np.isfinite(threshold))
        self.assertEqual(threshold, 4.0)


class _StubBand:
    def __init__(self, rec):
        self._rec = rec

    def WriteArray(self, arr):
        self._rec['written'] = arr

    def FlushCache(self):
        self._rec['flushed'] = True


class _StubDataset:
    def __init__(self, rec):
        self._rec = rec

    def SetGeoTransform(self, gt):
        self._rec['geotransform'] = gt

    def SetProjection(self, wkt):
        self._rec['projections'].append(wkt)

    def GetRasterBand(self, index):
        return _StubBand(self._rec)


class _StubDriver:
    def __init__(self, rec):
        self._rec = rec

    def Create(self, name, ncols, nrows, bands, dtype):
        self._rec['create_args'] = (name, ncols, nrows, bands, dtype)
        return _StubDataset(self._rec)


class _StubSRS:
    def __init__(self, rec, code):
        self._rec = rec
        self._code = code

    def GetAuthorityCode(self, target):
        return self._code

    def ImportFromEPSG(self, code):
        self._rec['epsg'] = code

    def ExportToWkt(self):
        return 'EPSG-WKT'


class FilteredDemCrsTests(ModuleStateTestCase):
    """F14: write_geotif_filteredDEM and the EPSG authority code.

    GetAuthorityCode('PROJCS') returns None for CRS without an EPSG code
    (custom/ESRI WKT); the old writer called int(None) -> TypeError. The
    end-to-end path needs GDAL bindings (not installed), so the writer is
    exercised through stub gdal/osr objects.
    """

    def write_filtered_dem(self, authority_code):
        rec = {'projections': [], 'epsg': 'not-called'}
        self.set_attr(rio, 'Parameters', types.SimpleNamespace(
            pmGrassGISfileName='/tmp/geonet-gn2-tests/filtered.tif',
            geotransform=(0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            inputwktInfo='source-wkt'))
        self.set_attr(rio.gdal, 'GDT_Float32', 'float32-stub')
        self.set_attr(rio.gdal, 'GetDriverByName',
                      lambda name: _StubDriver(rec))
        self.set_attr(rio.osr, 'SpatialReference',
                      lambda wkt: _StubSRS(rec, authority_code))
        rio.write_geotif_filteredDEM(np.ones((4, 4), dtype=np.float32),
                                     '/tmp/geonet-gn2-tests', 'filtered.tif')
        return rec

    def test_writer_tolerates_missing_epsg_authority(self):
        rec = self.write_filtered_dem(None)
        # no EPSG re-import attempted; the source projection set from
        # Parameters.inputwktInfo is kept
        self.assertEqual(rec['epsg'], 'not-called')
        self.assertEqual(rec['projections'], ['source-wkt'])

    def test_writer_reimports_projection_when_authority_present(self):
        rec = self.write_filtered_dem('32617')
        self.assertEqual(rec['epsg'], 32617)
        self.assertEqual(rec['projections'], ['source-wkt', 'EPSG-WKT'])


if __name__ == '__main__':
    unittest.main()
