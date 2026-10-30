"""Regression tests for the scikit-fmm negative-discriminant failure.

Reproduces the GeoNet fast-marching call shape
(pygeonet_fast_marching.Fast_Marching: phi zeros with 1 on flow cells and
-1 at the start point, speed carrying the reciprocal local cost, dx=0.01).

On scikit-fmm 2019.1.30 these inputs abort the whole march with an opaque
"Negative discriminant in time marcher quadratic." RuntimeError (upstream
scikit-fmm issue #18). On the pinned 2021.7.8 the march completes; the
domain guard converts any residual opaque failure into
FastMarchingDomainError with catchment context.

The heavy geospatial imports of the GeoNet modules (gdal, rasterio,
matplotlib, numba, dask, psutil) are stubbed so the test only needs
numpy + skfmm + pytest.
"""
import os
import sys
import types

import numpy as np
import pytest

# Make the repo root importable regardless of pytest rootdir.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub_modules():
    """Stub GeoNet's heavy geospatial dependencies before import."""
    stubs = {}

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
    prepare.Parameters = prepare
    plain_module('pygeonet_defaults')

    stubs.update(osgeo=osgeo, rasterio=rasterio, matplotlib=matplotlib,
                 numba=numba, psutil=psutil, dask=dask,
                 pygeonet_prepare=prepare)
    return stubs


_install_stub_modules()

import skfmm  # noqa: E402
import pygeonet_fast_marching as fmm  # noqa: E402


def build_geonet_style_inputs(shape, cost, start, flow_mask):
    """Build phi/speed exactly like Fast_Marching does per catchment."""
    phi = np.zeros(shape)
    speed = np.zeros(shape)
    phi[flow_mask] = 1
    speed[flow_mask] = cost[flow_mask]
    phi[start] = -1
    return phi, speed


# The reproduced failing input: rough (high local contrast) speed field on a
# half-domain basin. On skfmm 2019.1.30 this raises
# RuntimeError("Negative discriminant in time marcher quadratic.").
REPRO_SHAPE = (200, 200)


def build_repro_cost():
    yy, xx = np.mgrid[0:REPRO_SHAPE[0], 0:REPRO_SHAPE[1]]
    flow_mask = xx < REPRO_SHAPE[1] // 2
    cost = np.random.RandomState(0).uniform(
        0.1, 1.0, size=REPRO_SHAPE)
    return cost, flow_mask, (REPRO_SHAPE[0] // 2, 0)


def test_repro_input_completes_on_pinned_skfmm():
    """The reproduced negative-discriminant input completes the march."""
    cost, flow_mask, start = build_repro_cost()
    phi, speed = build_geonet_style_inputs(REPRO_SHAPE, cost, start, flow_mask)
    travel_time = skfmm.travel_time(phi, speed, dx=.01)
    assert np.all(np.isfinite(travel_time[flow_mask]))
    # the march completed and produced a bounded, positive travel-time field
    assert 0 < np.nanmax(travel_time) < np.inf


def test_guard_wraps_opaque_skfmm_failure_with_context():
    """Opaque skfmm RuntimeError becomes FastMarchingDomainError with context."""
    cost, flow_mask, start = build_repro_cost()
    phi, speed = build_geonet_style_inputs(REPRO_SHAPE, cost, start, flow_mask)

    def failing_travel_time(*args, **kwargs):
        raise RuntimeError(
            'Negative discriminant in time marcher quadratic.')

    original = skfmm.travel_time
    skfmm.travel_time = failing_travel_time
    try:
        with pytest.raises(fmm.FastMarchingDomainError) as excinfo:
            fmm.travel_time_with_domain_guard(
                phi, speed, dx=.01, catchmentIndex=7, startPoint=start)
    finally:
        skfmm.travel_time = original
    message = str(excinfo.value)
    assert 'catchment 7' in message
    assert '({0}, {1})'.format(*start) in message
    assert 'Negative discriminant in time marcher quadratic.' in message
    # the underlying error stays chained, nothing is swallowed
    assert excinfo.value.__cause__ is not None


def test_guard_wraps_degenerate_phi_valueerror_with_context():
    """skfmm's degenerate-input ValueError is surfaced as a domain error."""
    shape = (10, 10)
    phi = np.ones(shape)
    speed = np.ones(shape)

    # no monkeypatch needed: phi has no zero contour, skfmm raises ValueError
    with pytest.raises(fmm.FastMarchingDomainError) as excinfo:
        fmm.travel_time_with_domain_guard(
            phi, speed, dx=.01, catchmentIndex=1, startPoint=(0, 0))
    assert 'catchment 1' in str(excinfo.value)
