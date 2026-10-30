# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep per-job wheel audits as strict as the previous combined audit."""
import hashlib
import subprocess
import sys
from pathlib import Path
import zipfile

import pytest

from tools.check_native_wheels import check_wheels


PLATFORMS = ("manylinux_2_17_x86_64", "macosx_10_12_x86_64", "win_amd64")


def make_wheel(directory, platform, abi="cp37-abi3", binary=True, license=True):
    path = directory / ("geonet_kernels-0.1.0-{}-{}.whl".format(abi, platform))
    with zipfile.ZipFile(str(path), "w") as archive:
        if binary:
            suffix = ".pyd" if platform == "win_amd64" else ".so"
            archive.writestr("geonet_kernels/geonet_kernels" + suffix, b"test binary")
        if license:
            archive.writestr("geonet_kernels-0.1.0.dist-info/licenses/License.txt",
                             "GNU GENERAL PUBLIC LICENSE")
    return path


@pytest.mark.parametrize("platform,family", zip(PLATFORMS, ("manylinux", "macosx", "win_amd64")))
def test_per_platform_audit_logs_path_and_hash(tmp_path, capsys, platform, family):
    wheel = make_wheel(tmp_path, platform)
    check_wheels(tmp_path, (family,))
    output = capsys.readouterr().out
    assert str(wheel) in output
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() in output


def test_default_audit_still_requires_all_platforms(tmp_path):
    make_wheel(tmp_path, PLATFORMS[0])
    with pytest.raises(AssertionError, match="missing wheel family: macosx"):
        check_wheels(tmp_path)
    for platform in PLATFORMS[1:]:
        make_wheel(tmp_path, platform)
    check_wheels(tmp_path)


def test_wrong_platform_and_empty_directory_fail(tmp_path):
    with pytest.raises(AssertionError, match="no wheels found"):
        check_wheels(tmp_path, ("win_amd64",))
    make_wheel(tmp_path, PLATFORMS[0])
    with pytest.raises(AssertionError, match="missing wheel family: win_amd64"):
        check_wheels(tmp_path, ("win_amd64",))


@pytest.mark.parametrize("changes,message", [
    ({"abi": "cp38-abi3"}, "expected Python 3.7 stable ABI"),
    ({"binary": False}, "native extension missing"),
    ({"license": False}, "GPL license text missing"),
])
def test_per_platform_audit_keeps_content_gates(tmp_path, changes, message):
    make_wheel(tmp_path, PLATFORMS[0], **changes)
    with pytest.raises(AssertionError, match=message):
        check_wheels(tmp_path, ("manylinux",))


def test_family_cli(tmp_path):
    make_wheel(tmp_path, PLATFORMS[0])
    script = Path(__file__).resolve().parents[1] / "tools/check_native_wheels.py"
    subprocess.run([sys.executable, str(script), str(tmp_path), "--family", "manylinux"],
                   check=True)
    invalid = subprocess.run([sys.executable, str(script), str(tmp_path), "--family", "unknown"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert invalid.returncode == 2
