# SPDX-License-Identifier: GPL-3.0-or-later
"""Test the oracle infrastructure, not parity of unfinished native kernels."""
import hashlib

import numpy as np
import pytest

import golden


def test_missing_bundle_skips_loudly(monkeypatch):
    monkeypatch.delenv("GEONET_GOLDEN_ROOT", raising=False)
    with pytest.raises(pytest.skip.Exception, match="A1 parity has NOT been verified"):
        golden.bundle_root()


def test_bad_configured_bundle_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("GEONET_GOLDEN_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        golden.bundle_root()


def test_hash_verification_detects_corruption(tmp_path):
    path = tmp_path / "fixture.tif"
    path.write_bytes(b"frozen bytes")
    record = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert golden.verified_path(tmp_path, record) == path
    path.write_bytes(b"changed bytes")
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        golden.verified_path(tmp_path, record)


def test_byte_comparison_preserves_nan_payloads_and_signed_zero():
    expected = np.array([0x7FC00001, 0], dtype=np.uint32).view(np.float32)
    golden.assert_array_bytes_equal(expected.copy(), expected)
    for bits in ([0x7FC00002, 0], [0x7FC00001, 0x80000000]):
        with pytest.raises(AssertionError, match="pixel bytes differ"):
            golden.assert_array_bytes_equal(np.array(bits, dtype=np.uint32).view(np.float32), expected)


@pytest.mark.parametrize("actual", [np.zeros((2, 1), dtype=np.float32), np.zeros(2, dtype=np.float64)])
def test_shape_and_dtype_are_part_of_the_contract(actual):
    with pytest.raises(AssertionError, match="mismatch"):
        golden.assert_array_bytes_equal(actual, np.zeros(2, dtype=np.float32))


def test_raster_comparison_checks_both_oracle_and_actual(tmp_path, monkeypatch):
    expected = tmp_path / "expected.tif"
    actual = tmp_path / "actual.tif"
    expected.write_bytes(b"expected TIFF bytes")
    actual.write_bytes(expected.read_bytes())
    record = {"path": expected.name, "sha256": golden.sha256_file(expected)}
    monkeypatch.setitem(golden.CASES, "test", {"outputs": {"heads": record}})
    golden.assert_raster_matches_golden(actual, tmp_path, "test", "heads")
    actual.write_bytes(b"different TIFF bytes")
    with pytest.raises(AssertionError, match="file bytes differ"):
        golden.assert_raster_matches_golden(actual, tmp_path, "test", "heads")
    expected.write_bytes(b"corrupted oracle")
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        golden.assert_raster_matches_golden(actual, tmp_path, "test", "heads")


@pytest.mark.parametrize("case_name", sorted(golden.CASES))
def test_archived_fixture_integrity(case_name):
    root = golden.bundle_root()
    case = golden.verify_case(root, case_name)
    dem = golden.load_input(root, case_name)
    assert dem.shape == (case["size"], case["size"])
    assert dem.dtype == np.dtype("float32")
    for suffix in ("geodesicDistance", "channelHeads"):
        raster = golden.load_output(root, case_name, case_name + "_" + suffix + ".tif")
        assert raster.shape == dem.shape
