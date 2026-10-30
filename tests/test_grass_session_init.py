"""Regression tests for GRASS session init version dispatch.

GeoNet's GRASS wrapper historically called the GRASS 7.8 signature
``gsetup.init(gisbase, gisdb, location, mapset)``. GRASS 8.x changed the
signature to ``init(path, location=None, mapset=None, *, grass_path=None)``
and raised ``TypeError: init() takes from 1 to 3 positional arguments but
4 were given``. These tests pin the correct call shape for both
generations by mocking ``grass.script.setup`` — no real GRASS
installation is needed, so they run in CI environments without GRASS.
"""
import os
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pygeonet_grass_session import init_grass_session, parse_grass_major_version

GISBASE = "/usr/lib/grass84"
GISDB = "/home/user/grassdata"
LOCATION = "geonet"
MAPSET = "PERMANENT"


class RecordingGsetup:
    """Stands in for grass.script.setup and records init() calls."""

    def __init__(self):
        self.calls = []

    def init(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "session"


class ParseVersionTests(unittest.TestCase):
    def test_grass8_version_output(self):
        self.assertEqual(parse_grass_major_version("8.4.1"), 8)

    def test_grass7_version_output(self):
        self.assertEqual(parse_grass_major_version("7.8.8"), 7)

    def test_version_with_banner_text(self):
        self.assertEqual(parse_grass_major_version("GRASS GIS 8.4.1"), 8)

    def test_unparsable_version_returns_none(self):
        self.assertIsNone(parse_grass_major_version("not a version"))


class InitDispatchTests(unittest.TestCase):
    def setUp(self):
        self.gsetup = RecordingGsetup()

    def test_grass8_init_call_shape(self):
        init_grass_session(self.gsetup, GISBASE, GISDB, LOCATION, MAPSET, 8)
        self.assertEqual(len(self.gsetup.calls), 1)
        args, kwargs = self.gsetup.calls[0]
        # GRASS 8: gisdbase first, installation dir via keyword-only grass_path
        self.assertEqual(args, (GISDB, LOCATION, MAPSET))
        self.assertEqual(kwargs, {"grass_path": GISBASE})

    def test_grass7_init_call_shape(self):
        init_grass_session(self.gsetup, GISBASE, GISDB, LOCATION, MAPSET, 7)
        self.assertEqual(len(self.gsetup.calls), 1)
        args, kwargs = self.gsetup.calls[0]
        # GRASS 7.8: legacy 4-argument positional signature
        self.assertEqual(args, (GISBASE, GISDB, LOCATION, MAPSET))
        self.assertEqual(kwargs, {})

    def test_unsupported_version_fails_with_clear_message(self):
        for major in (6, 9, None):
            with self.assertRaises(RuntimeError) as ctx:
                init_grass_session(self.gsetup, GISBASE, GISDB, LOCATION, MAPSET, major)
            self.assertIn("Unsupported GRASS GIS major version", str(ctx.exception))
        self.assertEqual(self.gsetup.calls, [])


@unittest.skipUnless(shutil.which("grass"), "GRASS GIS not installed")
class RealGrassSignatureTests(unittest.TestCase):
    """Smoke test against the installed GRASS (skipped when absent, e.g. CI).

    TypeError is raised before init() runs any logic when the call shape
    does not match the installed signature, so a non-TypeError outcome
    proves signature compatibility; other exceptions (e.g. ValueError for
    a nonexistent path) still mean the signature fits.
    """

    def test_installed_grass_accepts_dispatched_call_shape(self):
        proc = subprocess.run(["grass", "--config", "path"],
                              capture_output=True, text=True, check=True)
        gisbase = proc.stdout.strip()
        sys.path.append(os.path.join(gisbase, "etc", "python"))
        import grass.script.setup as gsetup  # noqa: PLC0415

        try:
            init_grass_session(gsetup, gisbase, "/nonexistent-grassdata",
                               LOCATION, MAPSET, 8)
        except TypeError:
            self.fail("installed gsetup.init rejected the GRASS 8 call shape")
        except ValueError:
            pass  # expected: the database path does not exist


if __name__ == "__main__":
    unittest.main()
