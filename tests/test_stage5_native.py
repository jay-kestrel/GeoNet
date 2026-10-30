# SPDX-License-Identifier: GPL-3.0-or-later
"""Selection parity without importing the legacy suite's global GDAL stubs."""
import ast
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from golden import assert_array_bytes_equal
import pygeonet_kernels as dispatch

native = pytest.importorskip("geonet_kernels")


def reference_function():
    """Execute the unchanged function body, not a rewritten approximation."""
    path = Path(__file__).parents[1] / "pygeonet_fast_marching.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "Fast_Marching_Start_Point_Identification")
    node.decorator_list = []
    tree.body = [node]
    namespace = {"np": np, "prange": range}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[node.name]


@pytest.fixture
def python_reference(monkeypatch):
    module = types.ModuleType("pygeonet_fast_marching")
    module.Fast_Marching_Start_Point_Identification = reference_function()
    monkeypatch.setitem(sys.modules, "pygeonet_fast_marching", module)
    return module.Fast_Marching_Start_Point_Identification


def both_modes(monkeypatch, args):
    outputs = []
    for mode in ("python", "rust", "auto"):
        monkeypatch.setenv("GEONET_KERNELS", mode)
        outputs.append(dispatch.identify_start_points(*args))
    for output in outputs[1:]:
        assert_array_bytes_equal(output, outputs[0])
    return outputs[0]


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                  np.uint8, np.uint16, np.uint32, np.uint64,
                                  np.float32, np.float64, np.bool_])
@pytest.mark.parametrize("shape,labels", [((0, 0), 0), ((4, 4), 0),
                                        ((4, 4), 1), ((1, 1), 1), ((1, 9), 4)])
def test_edge_cases(python_reference, monkeypatch, dtype, shape, labels):
    basin = np.ones(shape, dtype=dtype)
    args = (None, basin, np.zeros((2, labels), dtype=np.int64),
            float(max(basin.size, 1)), 0, 0.1, labels)
    both_modes(monkeypatch, args)


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "negative", "stride"])
def test_sparse_labels_and_layouts(python_reference, monkeypatch, dtype, layout):
    basin = (np.arange(300 * 12).reshape(300, 12) % 11 - 2).astype(dtype)
    if layout == "F":
        basin = np.asfortranarray(basin)
    elif layout == "negative":
        basin = basin[::-1, ::-1]
    elif layout == "stride":
        basin = basin[::2, ::3]
    both_modes(monkeypatch, (None, basin, None, float(basin.size), 10, 8.0, 13))


@pytest.mark.parametrize("minimum", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("threshold", [0.0, 25.0000025,
                                     np.nextafter(25.0000025, -np.inf), np.nan])
def test_strict_gates(python_reference, monkeypatch, minimum, threshold):
    both_modes(monkeypatch, (None, np.array([[1, 2, 2, 0]], dtype=np.int16),
                            None, 4.0, minimum, threshold, 3))


@pytest.mark.parametrize("pixels", [0.0, -0.0, -1.0, np.inf, np.nan])
def test_degenerate_area(python_reference, monkeypatch, pixels):
    with np.errstate(divide="ignore", invalid="ignore"):
        both_modes(monkeypatch, (None, np.array([[1.0, 0.0]]), None,
                                pixels, 0, 0.0, 2))


def test_nan_fractional_and_out_of_range_labels(python_reference, monkeypatch):
    basin = np.array([[np.nan, np.inf, -np.inf, 1.5, -2, 0, 1, 2, 100]])
    both_modes(monkeypatch, (None, basin, None, 9.0, 0, 0.1, 3))


@pytest.mark.parametrize("basin", [np.ones(3), np.ones((1, 1, 1)),
                                  np.ones((2, 2), dtype=complex),
                                  np.ones((2, 2), dtype=object)])
def test_invalid_arrays_fail_explicitly(basin):
    with pytest.raises(TypeError, match="two-dimensional"):
        native.identify_start_points(None, basin, None, 4.0, 0, 0.1, 1)


def test_negative_iteration_count():
    with pytest.raises(OverflowError):
        native.identify_start_points(None, np.ones((2, 2)), None, 4.0, 0, 0.1, -1)


def test_pipeline_routes_only_selection():
    source = (Path(__file__).parents[1] / "pygeonet_fast_marching.py").read_text()
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n.func for n in ast.walk(main) if isinstance(n, ast.Call)]
    assert any(isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
               and f.value.id == "pygeonet_kernels" and f.attr == "identify_start_points"
               for f in calls)
    assert not any(isinstance(f, ast.Name)
                   and f.id == "Fast_Marching_Start_Point_Identification" for f in calls)
