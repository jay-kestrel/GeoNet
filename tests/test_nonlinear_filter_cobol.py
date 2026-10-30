"""Parity tests for the COBOL Perona-Malik nonlinear-filter port (spec D5).

Gate: rtol=1e-6, atol=2e-6 against the frozen reference oracle
(tests/reference_nonlinear_filter.py, pinned at 5e2e879) across the full
shape/option/niter matrix, layout variants, and NaN patterns - the same
matrix style as the Rust port's parity suite in pygeoflood. On this
toolchain the kernel is in fact bit-exact; the tolerance is the documented
gate, not a license to drift.

Routing-off default behavior must stay byte-identical (AC1), and a missing
kernel must raise rather than silently fall back (AC5).
"""
import hashlib
import os
import pathlib
import sys
import types
import unittest
from unittest import mock

import numpy as np

# Make the repo root importable regardless of the test runner's rootdir.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub_modules():
    """Stub GeoNet's heavy geospatial dependencies before import.

    Same stubbing as test_fast_marching.py; the filter module itself only
    needs numpy + scipy, but its import chain pulls rasterio/plot modules.
    """
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

    plain_module('pygeonet_prepare')
    plain_module('pygeonet_defaults')


_install_stub_modules()

import pygeonet_nonlinear_filter as nlf  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_nonlinear_filter import anisodiff as reference_anisodiff  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_PATH = os.path.join(
    REPO_ROOT, 'benchmarks', 'fixtures', 'synthetic_dem_128x128.npy')
FIXTURE_SHA256 = (
    'e56c7d5387a8b06265ec121fca8e3dbd31a0a5e46b511725c679e363d0a692b4')

# Parity gate (spec AC2 / locked decision 6): rtol=1e-6 class with a small
# absolute floor for near-zero fluxes.
RTOL = 1e-6
ATOL = 2e-6


def load_fixture():
    return np.load(FIXTURE_PATH)


def assert_parity(testcase, got, ref, msg=''):
    """NaN positions compare by mask equality, finite cells by tolerance."""
    testcase.assertEqual(got.dtype, np.float32,
                         'COBOL output must stay float32 (no promotion)')
    testcase.assertEqual(got.shape, ref.shape)
    nan_ref, nan_got = np.isnan(ref), np.isnan(got)
    testcase.assertTrue(
        np.array_equal(nan_ref, nan_got),
        f'NaN mask mismatch {msg}: {int(nan_ref.sum())} vs '
        f'{int(nan_got.sum())} NaN cells')
    testcase.assertTrue(
        np.array_equal(np.isinf(ref), np.isinf(got)),
        f'inf positions differ {msg}')
    fin = ~nan_ref
    if not fin.any():
        return  # all-NaN result: mask equality above is the full contract
    testcase.assertTrue(
        np.allclose(got[fin], ref[fin], rtol=RTOL, atol=ATOL),
        f'finite-cell parity mismatch {msg}; max abs diff '
        f'{np.max(np.abs(got[fin] - ref[fin])):.3e}')


def seeded_synthetic(shape, seed=7, lo=0.0, hi=300.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


class TestNonlinearFilterCobolParity(unittest.TestCase):
    """PYGEONET_COBOL=1 parity against the frozen reference."""

    def test_routing_off_bit_exact(self):
        """AC1: default path untouched - routing off is byte-identical to
        the frozen oracle."""
        img = load_fixture()
        ref = reference_anisodiff(img, 5, 2.5, 0.1, (1.0, 1.0), 2)
        self.assertNotIn('PYGEONET_COBOL', os.environ)
        got = nlf.anisodiff(img, 5, 2.5, 0.1, (1.0, 1.0), 2)
        self.assertTrue(np.array_equal(got.view(np.uint32),
                                       ref.view(np.uint32)),
                        'routing-off output changed - default behavior '
                        'must be byte-identical')

    def test_fixture_parity_default_option(self):
        """Main gate: committed fixture, option 2, iterated feedback."""
        img = load_fixture()
        ref = reference_anisodiff(img, 15, 2.5, 0.1, (1.0, 1.0), 2)
        with mock.patch.dict(os.environ, {'PYGEONET_COBOL': '1'}):
            got = nlf.anisodiff(img, 15, 2.5, 0.1, (1.0, 1.0), 2)
        assert_parity(self, got, ref, 'fixture option=2 niter=15')

    def test_fixture_parity_option1(self):
        """Main gate: committed fixture, option 1 (exp conductance)."""
        img = load_fixture()
        ref = reference_anisodiff(img, 15, 2.5, 0.1, (1.0, 1.0), 1)
        with mock.patch.dict(os.environ, {'PYGEONET_COBOL': '1'}):
            got = nlf.anisodiff(img, 15, 2.5, 0.1, (1.0, 1.0), 1)
        assert_parity(self, got, ref, 'fixture option=1 niter=15')

    def test_fixture_digest(self):
        """The committed workload may not drift silently."""
        with open(FIXTURE_PATH, 'rb') as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(digest, FIXTURE_SHA256)

    def test_shape_and_option_sweep(self):
        """Degenerate shapes, all options, zero/one/many iterations."""
        for shape in [(1, 1), (1, 7), (7, 1), (2, 2), (5, 9), (9, 5),
                      (7, 7), (16, 3)]:
            img = seeded_synthetic(shape, lo=-50.0)
            for option in (1, 2, 3):
                for niter in (0, 1, 5):
                    with self.subTest(shape=shape, option=option, niter=niter):
                        ref = reference_anisodiff(
                            img, niter, 2.5, 0.1, (1.0, 1.0), option)
                        with mock.patch.dict(os.environ,
                                             {'PYGEONET_COBOL': '1'}):
                            got = nlf.anisodiff(
                                img, niter, 2.5, 0.1, (1.0, 1.0), option)
                        assert_parity(
                            self, got, ref,
                            f'shape={shape} option={option} niter={niter}')

    def test_layout_variants(self):
        """Non-contiguous, transposed, strided, read-only, and foreign-dtype
        inputs produce identical results (adapter normalizes)."""
        base = seeded_synthetic((20, 14), seed=11, lo=-50.0)
        variants = {
            'fortran_order': np.asfortranarray(base),
            'transposed_view': base.T,
            'strided_slice': base[::2, ::3],
            'float64_input': base.astype(np.float64),
            'uint16_input': (base - 50.0).astype(np.uint16),
        }
        readonly = base[:10, :7].copy()
        readonly.setflags(write=False)
        variants['read_only_view'] = readonly
        for name, img in variants.items():
            with self.subTest(variant=name):
                ref = reference_anisodiff(img, 5, 2.5, 0.1, (1.0, 1.0), 2)
                with mock.patch.dict(os.environ, {'PYGEONET_COBOL': '1'}):
                    got = nlf.anisodiff(img, 5, 2.5, 0.1, (1.0, 1.0), 2)
                assert_parity(self, got, ref, name)

    def test_nan_patterns(self):
        """NaN inputs propagate identically; masks match position-for-position."""
        base = seeded_synthetic((13, 11), seed=13)

        def with_nan(pattern):
            img = base.copy()
            if pattern == 'corner':
                img[0, 0] = np.nan
            elif pattern == 'center':
                img[6, 5] = np.nan
            elif pattern == 'full_row':
                img[4, :] = np.nan
            elif pattern == 'full_col':
                img[:, 7] = np.nan
            elif pattern == 'scattered':
                img[(1, 2), (2, 9)] = np.nan
                img[11, 0] = np.nan
                img[3, 10] = np.nan
            elif pattern == 'all':
                img[:] = np.nan
            return img

        for pattern in ('corner', 'center', 'full_row', 'full_col',
                        'scattered', 'all'):
            img = with_nan(pattern)
            for option in (1, 2):
                with self.subTest(pattern=pattern, option=option):
                    ref = reference_anisodiff(
                        img, 5, 2.5, 0.1, (1.0, 1.0), option)
                    with mock.patch.dict(os.environ,
                                         {'PYGEONET_COBOL': '1'}):
                        got = nlf.anisodiff(
                            img, 5, 2.5, 0.1, (1.0, 1.0), option)
                    assert_parity(self, got, ref,
                                  f'{pattern} option={option}')

    def test_input_not_modified(self):
        """The adapter copies; the caller's array is untouched."""
        img = seeded_synthetic((13, 11), seed=17)
        orig = img.copy()
        with mock.patch.dict(os.environ, {'PYGEONET_COBOL': '1'}):
            nlf.anisodiff(img, 5, 2.5, 0.1, (1.0, 1.0), 2)
        self.assertTrue(np.array_equal(img, orig))

    def test_deterministic(self):
        """Repeated calls are bit-identical (kernel is stateless and
        single-threaded)."""
        img = seeded_synthetic((13, 11), seed=19)
        with mock.patch.dict(os.environ, {'PYGEONET_COBOL': '1'}):
            a1 = nlf.anisodiff(img, 5, 2.5, 0.1, (1.0, 1.0), 2)
            a2 = nlf.anisodiff(img, 5, 2.5, 0.1, (1.0, 1.0), 2)
        self.assertTrue(np.array_equal(a1.view(np.uint32),
                                       a2.view(np.uint32)))

    def test_missing_library_raises(self):
        """AC5: routing on + missing .so raises - never a silent fallback."""
        with mock.patch.dict(os.environ, {'PYGEONET_COBOL': '1'}), \
                mock.patch.object(
                    nlf, '_cobol_library_path',
                    return_value=pathlib.Path('/nonexistent/pygeonet_cobol.so')):
            with self.assertRaises(FileNotFoundError) as ctx:
                nlf.anisodiff(load_fixture(), 1, 2.5, 0.1)
        self.assertIn('make -C cobol', str(ctx.exception))

    def test_kappa_zero_raises(self):
        """Documented divergence: the reference emits inf with NumPy
        warnings; the COBOL backend rejects kappa=0."""
        with mock.patch.dict(os.environ, {'PYGEONET_COBOL': '1'}):
            with self.assertRaises(ValueError):
                nlf.anisodiff(seeded_synthetic((9, 7)), 3, 0.0, 0.1)


if __name__ == '__main__':
    unittest.main()
