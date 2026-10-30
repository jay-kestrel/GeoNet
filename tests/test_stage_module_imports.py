"""Import sweep: every pygeonet_*.py module must import on Python 3.11+.

The stage scripts are run as standalone files and nothing in the suite
imported stages 1, 2, 4, or 6, so import-time breakage there went
unnoticed: stage 6 imports time.clock (removed in Python 3.8) and stage
4 calls np.warnings.filterwarnings inside its threshold function
(removed in NumPy 2.0), each of which crashes the corresponding
pipeline stage outright on a modern interpreter. This sweep imports
every pygeonet_*.py module in the repo root on the CI interpreter and
runs the stage-4 threshold call on real NumPy, so an import-time or
NumPy-API regression in any stage fails CI immediately.

The heavy third-party dependencies (osgeo, rasterio, matplotlib, scipy,
statsmodels, numba, skfmm, psutil, dask) are stubbed so the sweep needs
only numpy and the standard library. NumPy itself stays real: its
module surface is exactly what broke stage 4 on NumPy 2.

pygeonet_prepare and pygeonet_defaults are the two exclusions: both
resolve the project cfg at import time and crash without one (a
behavior pinned for prepare by test_setup_scripts.py), so they are
stub-substituted like every other cfg-bound import in test_fast_marching.py.
"""
import glob
import importlib
import os
import sys
import types
import unittest

# Make the repo root importable regardless of the test runner's rootdir.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Modules that must never silently drop out of the sweep: the six
# pipeline stages plus the helpers they import.
REQUIRED_MODULES = frozenset({
    'pygeonet_nonlinear_filter',
    'pygeonet_slope_curvature',
    'pygeonet_grass_py3',
    'pygeonet_grass_session',
    'pygeonet_skeleton_definition',
    'pygeonet_fast_marching',
    'pygeonet_channel_head_definition',
    'pygeonet_rasterio',
    'pygeonet_vectorio',
    'pygeonet_plot',
    'pygeonet_check_setup',
    'pygeonet_envinfo',
    'pygeonet_configure',
})

# Both resolve the project cfg at import time and crash without one, so
# they cannot be imported hermetically (see module docstring).
CFG_BOUND_MODULES = frozenset({'pygeonet_prepare', 'pygeonet_defaults'})


def _install_stub_modules():
    """Stub GeoNet's heavy third-party dependencies before the sweep."""
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

    # rasterio + rasterio.crs + rasterio.features as used by
    # pygeonet_rasterio/pygeonet_skeleton_definition
    rasterio = plain_module('rasterio')
    rasterio_crs = plain_module('rasterio.crs')
    rasterio_crs.CRS = type('CRS', (), {})
    rasterio.crs = rasterio_crs
    rasterio.features = plain_module('rasterio.features')

    # matplotlib.pyplot / matplotlib.cm as used by pygeonet_plot and
    # pygeonet_slope_curvature (pygeonet_plot reads cm.coolwarm as a
    # default argument at import time)
    matplotlib = plain_module('matplotlib')
    matplotlib.pyplot = plain_module('matplotlib.pyplot')
    matplotlib.cm = plain_module('matplotlib.cm')
    matplotlib.cm.__getattr__ = lambda name: name
    matplotlib.use = lambda *a, **k: None

    # scipy submodules as used by the stage scripts
    scipy = plain_module('scipy')
    scipy.signal = plain_module('scipy.signal')
    scipy.ndimage = plain_module('scipy.ndimage')
    scipy_stats = plain_module('scipy.stats')
    scipy_stats.mstats = plain_module('scipy.stats.mstats')
    scipy_stats.mstats.mquantiles = lambda *a, **k: None
    scipy.stats = scipy_stats

    # statsmodels.api as used by pygeonet_slope_curvature
    statsmodels = plain_module('statsmodels')
    statsmodels.api = plain_module('statsmodels.api')

    # numba jit/njit/prange as used by the stage scripts
    numba = plain_module('numba')
    numba.jit = lambda *a, **k: (
        (lambda fn: fn) if not (a and callable(a[0])) else a[0])
    numba.njit = numba.jit
    numba.prange = range

    # skfmm as used by pygeonet_fast_marching
    plain_module('skfmm')

    # psutil.virtual_memory as used by Fast_Marching
    psutil = plain_module('psutil')
    psutil.virtual_memory = lambda: types.SimpleNamespace(total=0)

    # dask + dask.distributed.Client
    dask = plain_module('dask')
    dask.distributed = plain_module('dask.distributed')
    dask.distributed.Client = type('Client', (), {})

    # cfg-bound modules (see module docstring)
    plain_module('pygeonet_prepare')
    plain_module('pygeonet_defaults')


_install_stub_modules()


def discover_stage_modules():
    """Every pygeonet_*.py module in the repo root, minus cfg-bound ones."""
    names = sorted(
        os.path.basename(path)[:-3]
        for path in glob.glob(os.path.join(REPO_ROOT, 'pygeonet_*.py')))
    return [name for name in names if name not in CFG_BOUND_MODULES]


class StageModuleImportTests(unittest.TestCase):
    def test_required_modules_are_discovered(self):
        """The sweep cannot silently shrink below the known module set."""
        missing = REQUIRED_MODULES - set(discover_stage_modules())
        self.assertEqual(missing, set())

    def test_every_stage_module_imports_on_modern_python(self):
        """Every pygeonet_*.py module imports cleanly (see docstring)."""
        for name in discover_stage_modules():
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_skeleton_single_threshold_runs_on_numpy_2(self):
        """compute_skeleton_by_single_threshold runs on NumPy >= 2.

        np.warnings was removed in NumPy 2.0; the function's
        filterwarnings call must go through the stdlib warnings module.
        """
        import numpy as np  # noqa: E402  (real NumPy, never stubbed)

        import pygeonet_skeleton_definition as skeleton_definition

        dem = np.array([[0.0, 2.0], [3.0, 0.0]])
        result = skeleton_definition.compute_skeleton_by_single_threshold(
            dem, 1.0)
        np.testing.assert_array_equal(result, [[0.0, 1.0], [1.0, 0.0]])


if __name__ == '__main__':
    unittest.main()
