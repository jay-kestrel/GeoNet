# SPDX-License-Identifier: GPL-3.0-or-later
"""Byte-level full-stage parity on immutable e7e1bdb fixtures.

GEONET_GOLDEN_ROOT points to the extracted fl_NGrEVwuj archive.
GEONET_STAGE5_PYTHON points to its restored Python 3.7.7 environment with the
native wheel installed. Set GEONET_GOLDEN_LARGE=1 to include 4096 squared.
"""
import os

import pytest

from golden import bundle_root
from stage5_oracle import run_case, stage_environment


@pytest.mark.parametrize("parallel", [False, True])
def test_probe_thread_environment(monkeypatch, parallel):
    monkeypatch.setenv("NUMBA_NUM_THREADS", "3")
    monkeypatch.setenv("RAYON_NUM_THREADS", "3")
    env = stage_environment("rust", parallel)
    assert env["GEONET_KERNELS"] == "rust"
    for name in ("NUMBA_NUM_THREADS", "RAYON_NUM_THREADS"):
        assert env.get(name) == (None if parallel else "1")
        assert os.environ[name] == "3"


@pytest.mark.parametrize("case", ["bench_s_512", "bench_m_1024", "bench_l_4096"])
@pytest.mark.parametrize("mode", ["python", "rust"])
def test_stage5_tiff_bytes(case, mode, tmp_path):
    root = bundle_root()
    python = os.environ.get("GEONET_STAGE5_PYTHON")
    if not python:
        pytest.skip("GEONET_STAGE5_PYTHON absent: full stage-5 parity NOT verified")
    if case == "bench_l_4096" and os.environ.get("GEONET_GOLDEN_LARGE") != "1":
        pytest.skip("Set GEONET_GOLDEN_LARGE=1 for the 4096 squared acceptance run")
    report = run_case(root, tmp_path / case, case, python, mode=mode)
    assert report["records"][0]["byte_identical"]
    assert report["canonical_restored"]
