# SPDX-License-Identifier: GPL-3.0-or-later
"""Dispatch contracts without importing GeoNet's stateful I/O modules."""
import ast
import inspect
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest

import pygeonet_kernels as dispatch


CASES = [
    ("identify_start_points", "pygeonet_fast_marching",
     "Fast_Marching_Start_Point_Identification", 7),
    ("find_channel_heads", "pygeonet_channel_head_definition",
     "Channel_Head_Definition", 2),
]


@pytest.fixture(params=CASES)
def kernel(request, monkeypatch):
    public, module_name, original, count = request.param
    function = Mock(return_value=object())
    module = types.ModuleType(module_name)
    setattr(module, original, function)
    monkeypatch.setitem(sys.modules, module_name, module)
    return getattr(dispatch, public), function, tuple(object() for _ in range(count))


def absent_extension(name):
    raise ModuleNotFoundError("No module named 'geonet_kernels'", name=name)


@pytest.mark.parametrize("mode", ["python", "auto", None])
def test_python_fallback(kernel, monkeypatch, mode):
    function, original, args = kernel
    if mode is None:
        monkeypatch.delenv("GEONET_KERNELS", raising=False)
    else:
        monkeypatch.setenv("GEONET_KERNELS", mode)
    importer = Mock(side_effect=absent_extension)
    monkeypatch.setattr(dispatch.importlib, "import_module", importer)
    assert function(*args) is original.return_value
    original.assert_called_once_with(*args)
    assert importer.call_count == (0 if mode == "python" else 1)


def test_explicit_rust_never_falls_back(kernel, monkeypatch):
    function, original, args = kernel
    monkeypatch.setenv("GEONET_KERNELS", "rust")
    monkeypatch.setattr(dispatch.importlib, "import_module", absent_extension)
    with pytest.raises(ModuleNotFoundError, match="geonet_kernels"):
        function(*args)
    original.assert_not_called()


@pytest.mark.parametrize("mode", ["auto", "rust", "RuSt"])
def test_native_preferred_and_keyword_arguments_preserved(kernel, monkeypatch, mode):
    function, original, args = kernel
    native_function = Mock(return_value=object())
    native = types.SimpleNamespace(**{function.__name__: native_function})
    monkeypatch.setenv("GEONET_KERNELS", mode)
    monkeypatch.setattr(dispatch.importlib, "import_module", Mock(return_value=native))
    kwargs = dict(zip(inspect.signature(function).parameters, args))
    assert function(**kwargs) is native_function.return_value
    native_function.assert_called_once_with(*args)
    original.assert_not_called()


@pytest.mark.parametrize("error", [
    ImportError("broken extension binary"),
    ModuleNotFoundError("missing native dependency", name="numpy"),
])
def test_broken_import_is_not_hidden(kernel, monkeypatch, error):
    function, original, args = kernel
    monkeypatch.setenv("GEONET_KERNELS", "auto")
    monkeypatch.setattr(dispatch.importlib, "import_module", Mock(side_effect=error))
    with pytest.raises(type(error)) as caught:
        function(*args)
    assert caught.value is error
    original.assert_not_called()


def test_native_execution_error_is_not_hidden(kernel, monkeypatch):
    function, original, args = kernel
    native = types.SimpleNamespace(**{function.__name__: Mock(side_effect=ValueError("bad input"))})
    monkeypatch.setenv("GEONET_KERNELS", "auto")
    monkeypatch.setattr(dispatch.importlib, "import_module", Mock(return_value=native))
    with pytest.raises(ValueError, match="bad input"):
        function(*args)
    original.assert_not_called()


def test_invalid_mode_rejected(kernel, monkeypatch):
    function, original, args = kernel
    monkeypatch.setenv("GEONET_KERNELS", "typo")
    with pytest.raises(ValueError, match="GEONET_KERNELS"):
        function(*args)
    original.assert_not_called()


@pytest.mark.parametrize("public,module_name,original,count", CASES)
def test_signatures_match_unmodified_python_source(public, module_name, original, count):
    source = Path(__file__).resolve().parents[1] / (module_name + ".py")
    tree = ast.parse(source.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == original)
    names = [arg.arg for arg in node.args.args]
    assert list(inspect.signature(getattr(dispatch, public)).parameters) == names
    assert len(names) == count
