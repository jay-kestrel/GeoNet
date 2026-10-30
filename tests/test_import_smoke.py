"""Import-smoke test: import every Python-3 pygeonet_*.py module.

The adversarial review (F5) found that CI only verified extracted kernels:
no test imported pygeonet_nonlinear_filter, pygeonet_slope_curvature,
pygeonet_skeleton_definition, pygeonet_channel_head_definition,
pygeonet_vectorio, or pygeonet_rasterio -- which is exactly why the
``from time import clock`` (F1) and ``np.warnings`` (F2) breakages shipped
behind a green build. This test imports every py3-reachable pipeline module
so import-time incompatibilities fail CI permanently.

Heavy geospatial dependencies are stubbed the same way the existing test
modules do it: fall back to minimal ``osgeo`` / ``rasterio`` stubs when the
real bindings are not installed, and create a scratch project cfg when none
exists (pygeonet_prepare parses the cfg and creates the file structure at
import time).

``pygeonet_grass_py2.py`` is excluded: it is the Python-2-only GRASS script
(``import ConfigParser``, ``import grass.script``) and cannot import on any
Python 3 interpreter by design.
"""
import configparser
import glob
import importlib
import os
import shutil
import sys
import tempfile
import types
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Python-2-only scripts that cannot import on Python 3 by design.
PY2_ONLY_MODULES = ("pygeonet_grass_py2",)

_CFG_PATH = None
_CFG_HOME = None


def _cleanup_cfg():
    if _CFG_PATH and os.path.exists(_CFG_PATH):
        os.remove(_CFG_PATH)
    if _CFG_HOME and os.path.isdir(_CFG_HOME):
        shutil.rmtree(_CFG_HOME, ignore_errors=True)


def _ensure_import_env():
    """Stub osgeo/rasterio when absent; create a scratch cfg when none exists."""
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
    try:
        import rasterio  # noqa: F401
        import rasterio.features  # noqa: F401
        import rasterio.crs  # noqa: F401
    except ImportError:
        _rasterio = types.ModuleType("rasterio")
        _features = types.ModuleType("rasterio.features")
        sys.modules["rasterio.features"] = _features
        _rasterio.features = _features
        _crs = types.ModuleType("rasterio.crs")

        class _CRS:  # minimal stand-in; only identity is needed at import time
            pass

        _crs.CRS = _CRS
        sys.modules["rasterio.crs"] = _crs
        _rasterio.crs = _crs
        sys.modules["rasterio"] = _rasterio
    try:
        import statsmodels  # noqa: F401
        import statsmodels.api  # noqa: F401
    except ImportError:
        _statsmodels = types.ModuleType("statsmodels")
        _sm_api = types.ModuleType("statsmodels.api")
        sys.modules["statsmodels.api"] = _sm_api
        _statsmodels.api = _sm_api
        sys.modules["statsmodels"] = _statsmodels
    try:
        import dask  # noqa: F401
        import dask.distributed  # noqa: F401
    except ImportError:
        _dask = types.ModuleType("dask")
        _distributed = types.ModuleType("dask.distributed")
        _distributed.Client = object  # identity only; never used at import time
        sys.modules["dask.distributed"] = _distributed
        _dask.distributed = _distributed
        sys.modules["dask"] = _dask
    if glob.glob(os.path.join(REPO_ROOT, "GeoNet*.cfg")):
        return  # an existing user/project cfg wins; do not touch it
    _CFG_HOME = tempfile.mkdtemp(prefix="geonet-importsmoke-")
    _CFG_PATH = os.path.join(REPO_ROOT, "GeoNet_importsmoke.cfg")
    cfg = configparser.ConfigParser()
    cfg["Section"] = {
        "geofloodhomedir": _CFG_HOME,
        "projectname": "importsmoke",
        "dem_name": "dem",
        "Input_dir": "GeoInputs",
        "Output_dir": "GeoOutputs",
    }
    with open(_CFG_PATH, "w") as fh:
        cfg.write(fh)


_ensure_import_env()


def _ensure_real_numba():
    # Sibling test modules (test_fast_marching) leave a numba stub carrying
    # only njit/prange in sys.modules when they cannot find real numba.
    # pygeonet_channel_head_definition does ``from numba import jit``, so
    # when the real numba is importable, drop stub numba* entries and let
    # the genuine module load. Already-imported pygeonet modules keep their
    # bound references, so sibling tests are unaffected.
    for name in [n for n in list(sys.modules)
                 if n == "numba" or n.startswith("numba.")]:
        if getattr(sys.modules[name], "__spec__", None) is None:
            del sys.modules[name]
    try:
        import numba  # noqa: F401
    except ImportError:
        pass


class TestEveryModuleImports(unittest.TestCase):
    def test_every_pygeonet_module_imports(self):
        _ensure_real_numba()
        module_paths = sorted(
            glob.glob(os.path.join(REPO_ROOT, "pygeonet_*.py"))
        )
        self.assertGreater(len(module_paths), 0)
        for path in module_paths:
            name = os.path.splitext(os.path.basename(path))[0]
            if name in PY2_ONLY_MODULES:
                continue
            with self.subTest(module=name):
                importlib.import_module(name)


    def test_skeleton_threshold_runs_on_modern_numpy(self):
        # F2 regression: compute_skeleton_by_single_threshold called
        # np.warnings.filterwarnings (removed in numpy >= 1.22), crashing
        # stage 4 on every modern numpy. Pin the call itself, not just the
        # module import -- the broken line sits inside the function body.
        import numpy as np
        import pygeonet_skeleton_definition as skeleton

        result = skeleton.compute_skeleton_by_single_threshold(
            np.ones((4, 4)), 0.5
        )
        self.assertEqual(result.shape, (4, 4))


    def test_missing_cfg_exits_with_configure_message(self):
        # F11 regression: with no GeoNet_*.cfg and no project_pointer.cfg,
        # importing pygeonet_prepare used to die with a bare IndexError.
        # The real module runs in a scratch copy so this test does not
        # touch the repo root or any other test's cfg.
        import subprocess
        scratch = tempfile.mkdtemp(prefix="geonet-nocfg-")
        try:
            shutil.copy(os.path.join(REPO_ROOT, "pygeonet_prepare.py"), scratch)
            proc = subprocess.run(
                [sys.executable, "-c", "import pygeonet_prepare"],
                capture_output=True, text=True, cwd=scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("pygeonet_configure.py", proc.stderr)


if __name__ == "__main__":
    unittest.main()
