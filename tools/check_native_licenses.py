# SPDX-License-Identifier: GPL-3.0-or-later
"""Reject unknown/non-permissive dependency licenses in the resolved graph."""
import json
from pathlib import Path
import sys

# Exact expressions observed in the checked-in Cargo.lock's metadata. New
# expressions require explicit review rather than a substring-based bypass.
PERMISSIVE = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "MIT OR Apache-2.0",
    "Apache-2.0 OR MIT", "MIT/Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "(MIT OR Apache-2.0) AND Unicode-3.0",
}


def check_licenses(metadata):
    for package in metadata["packages"]:
        license_id = package["license"]
        if package["id"] in metadata["workspace_members"]:
            assert license_id == "GPL-3.0-or-later", package["name"]
        else:
            assert license_id in PERMISSIVE, "unreviewed dependency license: {} {}".format(
                package["name"], license_id)
        print(package["name"], package["version"], license_id)


if __name__ == "__main__":
    check_licenses(json.loads(Path(sys.argv[1]).read_text()))
