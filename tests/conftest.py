"""Pytest fixtures for GeoNet stage-5 regression tests.

GeoNet's modules read a project cfg at import time (pygeonet_prepare globs
``GeoNet*.cfg`` in the repo root) and import GDAL bindings from ``osgeo``.
For a self-sufficient test run on a fresh checkout, this conftest:

- puts the repo root on ``sys.path``,
- creates a scratch ``GeoNet_pytest.cfg`` (pointing ``geofloodhomedir`` at a
  session temp dir) when no cfg exists, and removes it afterwards,
- falls back to a minimal ``osgeo`` stub only when the GDAL bindings are not
  installed (the stage-5 numba kernel under test does no raster I/O).
"""
import configparser
import glob
import os
import shutil
import sys
import tempfile
import types

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(REPO_ROOT, "GeoNet_pytest.cfg")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import osgeo.gdal  # noqa: F401
except ImportError:
    _osgeo = types.ModuleType("osgeo")
    for _sub in ("gdal", "osr", "ogr"):
        _mod = types.ModuleType("osgeo.%s" % _sub)
        sys.modules["osgeo.%s" % _sub] = _mod
        setattr(_osgeo, _sub, _mod)
    sys.modules["osgeo"] = _osgeo

_created_cfg = None
_tmp_home = None


def pytest_configure(config):
    global _created_cfg, _tmp_home
    if glob.glob(os.path.join(REPO_ROOT, "GeoNet*.cfg")):
        return  # an existing user/project cfg wins; do not touch it
    _tmp_home = tempfile.mkdtemp(prefix="geonet-tests-")
    cfg = configparser.ConfigParser()
    cfg["Section"] = {
        "geofloodhomedir": _tmp_home,
        "projectname": "pytest",
        "dem_name": "dem",
        "Input_dir": "GeoInputs",
        "Output_dir": "GeoOutputs",
    }
    with open(CFG_PATH, "w") as fh:
        cfg.write(fh)
    _created_cfg = CFG_PATH


def pytest_unconfigure(config):
    if _created_cfg and os.path.exists(_created_cfg):
        os.remove(_created_cfg)
    if _tmp_home and os.path.isdir(_tmp_home):
        shutil.rmtree(_tmp_home, ignore_errors=True)


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT
