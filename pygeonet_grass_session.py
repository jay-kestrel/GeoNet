"""GRASS session initialization helpers shared by the GeoNet GRASS wrapper.

GRASS GIS 7.8 and 8.x expose incompatible ``grass.script.setup.init``
signatures:

    GRASS 7.8: init(gisbase, gisdb, location, mapset)
    GRASS 8.x: init(path, location=None, mapset=None, *, grass_path=None)

where the GRASS 8 ``path`` argument is the GISDBASE and the installation
directory moved to the keyword-only ``grass_path`` argument. These helpers
dispatch on the installed major version so GeoNet works with either
series, and fail with a clear message on anything else.
"""


def parse_grass_major_version(version_string):
    """Return the GRASS major version as an int, or None if unparsable.

    Accepts the output of ``grass --config version`` (e.g. ``8.4.1`` or
    ``7.8.8``), possibly with surrounding banner text such as
    ``GRASS GIS 8.4.1``.
    """
    for token in str(version_string).replace("GRASS GIS", " ").split():
        parts = token.split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0])
    return None


def init_grass_session(gsetup, gisbase, gisdb, location, mapset, grass_major):
    """Initialize a GRASS session with the call shape of ``grass_major``.

    ``gsetup`` is the ``grass.script.setup`` module of the installed
    GRASS; the remaining arguments mirror the legacy GRASS 7.8
    ``gsetup.init`` call. Raises RuntimeError for GRASS versions GeoNet
    does not support (anything other than major versions 7 and 8).
    """
    if grass_major == 8:
        return gsetup.init(gisdb, location, mapset, grass_path=gisbase)
    if grass_major == 7:
        return gsetup.init(gisbase, gisdb, location, mapset)
    raise RuntimeError(
        "Unsupported GRASS GIS major version %r: GeoNet supports "
        "GRASS GIS 7.8 and 8.x session initialization only." % (grass_major,)
    )
