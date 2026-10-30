# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage-6 native kernel: golden parity (A1), edge equivalence (A2), routing.

A1: the native find_channel_heads reproduces the archived e7e1bdb one-thread
channel-head rasters byte-for-byte on the frozen v6 inputs (512/1024/4096).

A2: on synthetic edge inputs the native path and the untouched Python
Channel_Head_Definition agree exactly -- same outputs on qualifying cases,
same IndexError on the preserved failure modes (empty mask: no components;
single component: the auto threshold always exceeds the only component's
count, so the scan finds zero heads and the return-array build raises).

Only numpy + tifffile are real for the golden tests; every heavy geospatial
dependency is stubbed (the same pattern as test_stage5_scatter.py) so the
Python oracle module can import.
"""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import golden  # noqa: E402
import geonet_kernels  # noqa: E402  (maturin-built extension: hard requirement)
import pygeonet_kernels as dispatch  # noqa: E402


def _install_stub_modules():
    """Stub GeoNet's heavy geospatial dependencies before import."""
    def plain_module(name):
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module

    osgeo = plain_module('osgeo')
    for submodule in ('gdal', 'osr', 'ogr'):
        sub = types.ModuleType('osgeo.' + submodule)
        setattr(osgeo, submodule, sub)
        sys.modules['osgeo.' + submodule] = sub

    rasterio = plain_module('rasterio')
    rasterio_crs = plain_module('rasterio.crs')
    rasterio_crs.CRS = type('CRS', (), {})
    rasterio.crs = rasterio_crs

    matplotlib = plain_module('matplotlib')
    matplotlib.pyplot = plain_module('matplotlib.pyplot')
    matplotlib.cm = plain_module('matplotlib.cm')
    matplotlib.cm.__getattr__ = lambda name: name
    matplotlib.use = lambda *a, **k: None

    # pygeonet_channel_head_definition does ``from numba import jit``;
    # pygeonet_fast_marching (via the dispatch shim) needs njit and prange.
    numba = plain_module('numba')
    def _identity_decorator(*a, **k):
        return (lambda fn: fn) if not (a and callable(a[0])) else a[0]
    numba.jit = _identity_decorator
    numba.njit = _identity_decorator
    numba.prange = range

    plain_module('psutil')
    psutil = sys.modules['psutil']
    psutil.virtual_memory = lambda: types.SimpleNamespace(total=0)

    if 'pygeonet_prepare' not in sys.modules:
        plain_module('pygeonet_prepare')
    defaults = plain_module('pygeonet_defaults')
    # Constants the stage-6 function (and the native kernel) read.
    defaults.doPlot = 0
    defaults.endPointSearchBoxSize = 30


_install_stub_modules()

import pygeonet_channel_head_definition as ch  # noqa: E402


GOLDEN_CASES = ("bench_s_512", "bench_m_1024", "bench_l_4096")


@pytest.mark.parametrize("case_name", GOLDEN_CASES)
def test_golden_channel_heads_byte_match(case_name):
    """A1: native heads == archived e7e1bdb one-thread raster, byte-for-byte."""
    root = golden.bundle_root()
    golden.verify_case(root, case_name)
    skeleton = golden.load_output(root, case_name, case_name + "_skeleton.tif")
    dist = golden.load_output(root, case_name, case_name + "_geodesicDistance.tif")
    expected = golden.load_output(root, case_name, case_name + "_channelHeads.tif")

    xx, yy = geonet_kernels.find_channel_heads(skeleton, dist)

    raster = np.zeros(dist.shape)
    raster[np.asarray(yy), np.asarray(xx)] = 1
    golden.assert_array_bytes_equal(raster.astype(np.float32), expected)


def test_golden_head_count_matches_upstream():
    """Upstream archived head counts are 2 / 2 / 4 (512/1024/4096)."""
    root = golden.bundle_root()
    for case_name, expected in zip(GOLDEN_CASES, (2, 2, 4)):
        golden.verify_case(root, case_name)
        skeleton = golden.load_output(root, case_name, case_name + "_skeleton.tif")
        dist = golden.load_output(root, case_name, case_name + "_geodesicDistance.tif")
        xx, yy = geonet_kernels.find_channel_heads(skeleton, dist)
        assert len(np.asarray(xx)) == expected


# --- A2: edge equivalence between the native and Python paths -------------


def _empty_mask_case():
    return (np.zeros((16, 16), dtype=np.float32),
            np.zeros((16, 16), dtype=np.float32))


def _single_component_case():
    skeleton = np.zeros((40, 40), dtype=np.float32)
    dist = np.zeros((40, 40), dtype=np.float32)
    for row in range(2, 8):
        skeleton[row, 2] = 1.0
        dist[row, 2] = float(8 - row)
    return skeleton, dist


def _seeded_random_case():
    rng = np.random.default_rng(20261030)
    field = rng.standard_normal((60, 60)).astype(np.float32)
    skeleton = (field > 1.0).astype(np.float32)
    dist = np.abs(rng.standard_normal((60, 60))).astype(np.float32)
    return skeleton, dist


@pytest.mark.parametrize("case_builder", [_empty_mask_case, _single_component_case])
def test_edge_inputs_raise_indexerror_in_both_paths(case_builder, monkeypatch):
    """Preserved failure modes: empty mask and single-component inputs."""
    skeleton, dist = case_builder()
    with pytest.raises(IndexError):
        ch.Channel_Head_Definition(skeleton, dist)
    with pytest.raises(IndexError):
        geonet_kernels.find_channel_heads(skeleton, dist)


def test_seeded_random_output_matches_python(monkeypatch):
    """Native heads == Python heads, coordinate-for-coordinate, on a fixed
    multi-component synthetic skeleton (writes stubbed to no-ops)."""
    monkeypatch.setattr(ch, "write_drainage_nodes", lambda *a, **k: None)
    monkeypatch.setattr(ch, "write_geotif_generic", lambda *a, **k: None)
    skeleton, dist = _seeded_random_case()
    xx_py, yy_py = ch.Channel_Head_Definition(skeleton.copy(), dist.copy())
    xx_rs, yy_rs = geonet_kernels.find_channel_heads(skeleton, dist)
    assert np.asarray(xx_rs).dtype == xx_py.dtype
    golden.assert_array_bytes_equal(
        np.asarray(xx_rs)[None, :], np.asarray(xx_py, dtype=xx_rs.dtype)[None, :])
    golden.assert_array_bytes_equal(
        np.asarray(yy_rs)[None, :], np.asarray(yy_py, dtype=yy_rs.dtype)[None, :])


def test_seeded_random_raster_matches_python(monkeypatch):
    """The assembled head raster equals the Python path's assembled raster."""
    monkeypatch.setattr(ch, "write_drainage_nodes", lambda *a, **k: None)
    monkeypatch.setattr(ch, "write_geotif_generic", lambda *a, **k: None)
    skeleton, dist = _seeded_random_case()
    xx_py, yy_py = ch.Channel_Head_Definition(skeleton.copy(), dist.copy())
    xx_rs, yy_rs = geonet_kernels.find_channel_heads(skeleton, dist)
    py_raster = np.zeros(dist.shape)
    py_raster[np.asarray(yy_py), np.asarray(xx_py)] = 1
    rs_raster = np.zeros(dist.shape)
    rs_raster[np.asarray(yy_rs), np.asarray(xx_rs)] = 1
    golden.assert_array_bytes_equal(rs_raster.astype(np.float32),
                                    py_raster.astype(np.float32))


# --- routing: the stage module's main() goes through the dispatcher -------


def test_main_routes_native(monkeypatch):
    skeleton, dist = _single_component_case()
    reads = iter([(skeleton,), (dist,)])
    monkeypatch.setattr(ch, "read_geotif_generic", lambda *a: next(reads))
    xx, yy = np.array([3, 5]), np.array([2, 7])
    calls = {}

    def fake_kernel(*args):
        calls["args"] = args
        return xx, yy

    def fake_writer(skel, dst, out_xx, out_yy):
        calls["written"] = (skel, dst, out_xx, out_yy)

    monkeypatch.setattr(dispatch, "_native_module", lambda: object())
    monkeypatch.setattr(dispatch, "find_channel_heads", fake_kernel)
    monkeypatch.setattr(ch, "_write_channel_head_outputs", fake_writer)
    ch.main()
    assert calls["args"] == (skeleton, dist)
    assert calls["written"] == (skeleton, dist, xx, yy)


def test_main_routes_python_fallback(monkeypatch):
    skeleton, dist = _single_component_case()
    reads = iter([(skeleton,), (dist,)])
    monkeypatch.setattr(ch, "read_geotif_generic", lambda *a: next(reads))
    monkeypatch.setattr(dispatch, "_native_module", lambda: None)
    routed = {}
    monkeypatch.setattr(ch, "Channel_Head_Definition",
                        lambda *a: routed.setdefault("args", a) or (None, None))
    ch.main()
    assert routed["args"] == (skeleton, dist)
