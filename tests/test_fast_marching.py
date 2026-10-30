"""Regression tests for the scikit-fmm negative-discriminant failure.

Reproduces the GeoNet fast-marching call shape
(pygeonet_fast_marching.Fast_Marching: phi zeros with 1 on flow cells and
-1 at the start point, speed carrying the reciprocal local cost, dx=0.01).

On scikit-fmm 2019.1.30 these inputs abort the whole march with an opaque
"Negative discriminant in time marcher quadratic." RuntimeError (upstream
scikit-fmm issue #18). On the pinned 2021.7.8 (and any release >= 2021.1.20)
the march completes; the domain guard converts any residual opaque failure
into FastMarchingDomainError with catchment context.

The heavy geospatial imports of the GeoNet modules (gdal, rasterio,
matplotlib, numba, dask, psutil) are stubbed so the test only needs
numpy + skfmm + the standard library.
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

    # numba njit/prange as used by pygeonet_fast_marching
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

    # pygeonet_prepare parses the project cfg at import time and crashes
    # without one; the domain guard under test needs none of it.
    prepare = plain_module('pygeonet_prepare')
    plain_module('pygeonet_defaults')


_install_stub_modules()

import numpy as np  # noqa: E402
import skfmm  # noqa: E402
import pygeonet_fast_marching as fmm  # noqa: E402

# The reproduced failing input: rough (high local contrast) speed field on a
# half-domain basin. On skfmm 2019.1.30 this raises
# RuntimeError("Negative discriminant in time marcher quadratic.").
REPRO_SHAPE = (200, 200)


def build_repro_inputs():
    """Build the phi/speed pair exactly like Fast_Marching does per catchment."""
    yy, xx = np.mgrid[0:REPRO_SHAPE[0], 0:REPRO_SHAPE[1]]
    flow_mask = xx < REPRO_SHAPE[1] // 2
    cost = np.random.RandomState(0).uniform(0.1, 1.0, size=REPRO_SHAPE)
    phi = np.zeros(REPRO_SHAPE)
    speed = np.zeros(REPRO_SHAPE)
    phi[flow_mask] = 1
    speed[flow_mask] = cost[flow_mask]
    start = (REPRO_SHAPE[0] // 2, 0)
    phi[start] = -1
    return phi, speed, flow_mask, start


class ReproInputCompletesOnPinnedSkfmmTests(unittest.TestCase):
    def test_rough_cost_field_completes_the_march(self):
        """The reproduced negative-discriminant input completes the march."""
        phi, speed, flow_mask, start = build_repro_inputs()
        travel_time = skfmm.travel_time(phi, speed, dx=.01)
        self.assertTrue(np.all(np.isfinite(travel_time[flow_mask])))
        # the march completed and produced a bounded, positive field
        self.assertGreater(float(np.nanmax(travel_time)), 0.0)
        self.assertLess(float(np.nanmax(travel_time)), float('inf'))


class GuardWrapsFailuresWithContextTests(unittest.TestCase):
    def test_opaque_runtime_error_becomes_domain_error(self):
        """Opaque skfmm RuntimeError becomes FastMarchingDomainError."""
        phi, speed, flow_mask, start = build_repro_inputs()

        def failing_travel_time(*args, **kwargs):
            raise RuntimeError(
                'Negative discriminant in time marcher quadratic.')

        original = skfmm.travel_time
        skfmm.travel_time = failing_travel_time
        try:
            with self.assertRaises(fmm.FastMarchingDomainError) as ctx:
                fmm.travel_time_with_domain_guard(
                    phi, speed, dx=.01, catchmentIndex=7, startPoint=start)
        finally:
            skfmm.travel_time = original
        message = str(ctx.exception)
        self.assertIn('catchment 7', message)
        self.assertIn('({0}, {1})'.format(*start), message)
        self.assertIn('Negative discriminant in time marcher quadratic.',
                      message)
        # the underlying error stays chained, nothing is swallowed
        self.assertIsNotNone(ctx.exception.__cause__)

    def test_degenerate_phi_valueerror_becomes_domain_error(self):
        """skfmm's degenerate-input ValueError is surfaced as a domain error."""
        phi = np.ones((10, 10))
        speed = np.ones((10, 10))
        # no monkeypatch needed: phi has no zero contour, skfmm raises ValueError
        with self.assertRaises(fmm.FastMarchingDomainError) as ctx:
            fmm.travel_time_with_domain_guard(
                phi, speed, dx=.01, catchmentIndex=1, startPoint=(0, 0))
        self.assertIn('catchment 1', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
