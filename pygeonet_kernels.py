# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in dispatch boundary; no pipeline call-sites use this scaffold yet.

GEONET_KERNELS is read on each call. Auto falls back only when the extension
itself is absent; broken native imports and native execution errors propagate.
Installed scaffold functions deliberately panic until the kernel ports land.
"""
import importlib
import os


def _native_module():
    mode = os.environ.get("GEONET_KERNELS", "auto").lower()
    if mode not in ("auto", "rust", "python"):
        raise ValueError("GEONET_KERNELS must be auto, rust, or python; got {!r}".format(mode))
    if mode == "python":
        return None
    try:
        return importlib.import_module("geonet_kernels")
    except ModuleNotFoundError as exc:
        if mode == "auto" and exc.name == "geonet_kernels":
            return None
        raise


def identify_start_points(outlet_array, basinIndexArray,
                          fastMarchingStartPointList, nDempixels,
                          basin_elements, threshold, iter_total):
    """Forward the original stage-5 arguments without copies or coercion."""
    native = _native_module()
    if native is not None:
        function = native.identify_start_points
    else:
        from pygeonet_fast_marching import Fast_Marching_Start_Point_Identification
        function = Fast_Marching_Start_Point_Identification
    return function(outlet_array, basinIndexArray, fastMarchingStartPointList,
                    nDempixels, basin_elements, threshold, iter_total)


def find_channel_heads(skeletonFromFlowAndCurvatureArray, geodesicDistanceArray):
    """Forward the original two-array stage-6 interface."""
    native = _native_module()
    if native is not None:
        function = native.find_channel_heads
    else:
        from pygeonet_channel_head_definition import Channel_Head_Definition
        function = Channel_Head_Definition
    return function(skeletonFromFlowAndCurvatureArray, geodesicDistanceArray)
