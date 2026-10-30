# SPDX-License-Identifier: GPL-3.0-or-later
"""Inspect installed-package contents without executing wheel code."""
import argparse
import hashlib
from pathlib import Path
import zipfile


WHEEL_FAMILIES = ("manylinux", "macosx", "win_amd64")


def check_wheel(path):
    assert "-cp37-abi3-" in path.name, "expected Python 3.7 stable ABI: {}".format(path)
    with zipfile.ZipFile(str(path)) as archive:
        names = archive.namelist()
        assert any(name.startswith("geonet_kernels/") and name.endswith((".so", ".pyd"))
                   for name in names), "native extension missing: {}".format(path)
        licenses = [name for name in names if ".dist-info/licenses/" in name]
        assert any(b"GNU GENERAL PUBLIC LICENSE" in archive.read(name)
                   for name in licenses), "GPL license text missing: {}".format(path)
    print(path, "native module + cp37 ABI + GPL verified")
    print("SHA256 {}  {}".format(hashlib.sha256(path.read_bytes()).hexdigest(), path))


def check_wheels(directory, families=WHEEL_FAMILIES):
    wheels = sorted(Path(directory).glob("*.whl"))
    assert wheels, "no wheels found"
    for family in families:
        assert any(family in path.name for path in wheels), "missing wheel family: " + family
    for path in wheels:
        check_wheel(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--family", choices=WHEEL_FAMILIES, action="append",
                        help="Required platform family; default: all three platforms")
    args = parser.parse_args()
    check_wheels(args.directory, args.family or WHEEL_FAMILIES)
