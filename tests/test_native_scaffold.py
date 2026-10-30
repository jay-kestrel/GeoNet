# SPDX-License-Identifier: GPL-3.0-or-later
"""Installed-wheel smoke tests; deliberately do not execute unfinished kernels."""
import inspect

import pytest

native = pytest.importorskip("geonet_kernels", reason="native wheel not installed; build with maturin first")


@pytest.mark.parametrize("name,parameters", [
    ("identify_start_points", ["outlet_array", "basinIndexArray",
     "fastMarchingStartPointList", "nDempixels", "basin_elements", "threshold", "iter_total"]),
    ("find_channel_heads", ["skeletonFromFlowAndCurvatureArray", "geodesicDistanceArray"]),
])
def test_exported_signatures(name, parameters):
    function = getattr(native, name)
    assert list(inspect.signature(function).parameters) == parameters
    with pytest.raises(TypeError):
        function()


def test_only_two_public_functions():
    names = {name for name in dir(native)
             if not name.startswith("_") and callable(getattr(native, name))}
    assert names == {"identify_start_points", "find_channel_heads"}
