# SPDX-License-Identifier: GPL-3.0-or-later
"""Hash-pinned upstream e7e1bdb oracle. Never regenerates expected outputs."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

MANIFEST = json.loads(Path(__file__).with_name("golden_manifest.json").read_text())
CASES = MANIFEST["cases"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_root():
    """Absence is a visible skip; a configured but broken bundle is a failure."""
    configured = os.environ.get("GEONET_GOLDEN_ROOT")
    if not configured:
        pytest.skip("upstream golden bundle fl_NGrEVwuj absent: extract it and set "
                    "GEONET_GOLDEN_ROOT; A1 parity has NOT been verified")
    root = Path(configured)
    if not root.is_dir():
        raise FileNotFoundError("GEONET_GOLDEN_ROOT is not a directory: {}".format(root))
    return root


def verified_path(root, record):
    path = Path(root) / record["path"]
    actual = sha256_file(path)
    if actual != record["sha256"]:
        raise AssertionError("golden SHA-256 mismatch for {}: {} != {}".format(
            path, actual, record["sha256"]))
    return path


def verify_case(root, case_name):
    """Verify input and every archived raster before use as an oracle."""
    case = CASES[case_name]
    verified_path(root, case["input"])
    for record in case["outputs"].values():
        verified_path(root, record)
    return case


def _read_raster(path):
    # No GDAL/project imports: the legacy unittest suite stubs those globally.
    import tifffile
    return tifffile.imread(str(path))


def load_input(root, case_name):
    return _read_raster(verified_path(root, CASES[case_name]["input"]))


def load_output(root, case_name, filename):
    return _read_raster(verified_path(root, CASES[case_name]["outputs"][filename]))


def assert_array_bytes_equal(actual, expected):
    """Reject even NaN payload or signed-zero differences; no float tolerance."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, "raster shape mismatch"
    assert actual.dtype == expected.dtype, "raster dtype mismatch"
    assert actual.tobytes(order="C") == expected.tobytes(order="C"), "raster pixel bytes differ"


def assert_raster_matches_golden(actual_path, root, case_name, filename):
    """Require exact TIFF bytes (including metadata), not merely similar pixels."""
    record = CASES[case_name]["outputs"][filename]
    verified_path(root, record)
    assert sha256_file(actual_path) == record["sha256"], "raster file bytes differ: {}".format(actual_path)
