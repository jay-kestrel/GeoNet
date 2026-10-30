#! /usr/bin/env python
"""Pre-flight checks for the GeoNet/GeoFlood pipeline.

    python pygeonet_check_setup.py [--imports]

Verifies, with readable messages and a non-zero exit code on failure,
everything the pipeline stages need before the 5-7 minute early stages
run:

* the project cfg resolves exactly the way pygeonet_prepare.py resolves
  it at import time (project_pointer.cfg pointer first, then a
  GeoNet*.cfg glob next to this script) and parses with the five keys
  prepare reads,
* the GeoNet home directory exists,
* the input DEM (cfg dem_name + '.tif') exists at the path the pipeline
  reads it from, <home>/<Input_dir>/GIS/<project>,
* the project directories exist (missing ones are warnings, because
  pygeonet_prepare.py creates them on import).

With --imports, additionally probe every third-party module the pipeline
imports at module level and fail on the ones that cannot be imported,
reporting the actual import error behind failures like broken GDAL
installations.

Why: pygeonet_prepare.py resolves the cfg and creates directories at
import time, so a missing cfg surfaces as a bare IndexError (the
glob(...)[0] fallback in pygeonet_prepare.py) and a missing DEM only
fails later, inside GDAL, after the pipeline has already started.

Stdlib-only at module level and no import-time side effects: safe to
import, and safe to run from anywhere (cfg resolution is relative to
this script's directory, matching pygeonet_configure.py).
"""
import argparse
import configparser
import glob
import importlib
import os
import sys

CFG_SECTION = 'Section'
# The five keys pygeonet_prepare.py reads from the project cfg.
CFG_KEYS = ('geofloodhomedir', 'projectname', 'dem_name',
            'Input_dir', 'Output_dir')

# Third-party modules the pipeline imports at module level
# (pygeonet_rasterio.py, pygeonet_plot.py, pygeonet_nonlinear_filter.py,
# pygeonet_fast_marching.py, pygeonet_skeleton_definition.py,
# pygeonet_slope_curvature.py, pygeonet_channel_head_definition.py,
# pygeonet_grass_py3.py).
PIPELINE_IMPORTS = (
    'numpy', 'scipy', 'matplotlib', 'osgeo.gdal', 'rasterio',
    'numba', 'skfmm', 'psutil', 'dask.distributed', 'statsmodels',
    'grass.script',
)

OK = 'ok'
WARN = 'warn'
FAIL = 'fail'


class SetupError(Exception):
    """A setup problem with a user-facing message."""


def resolve_cfg_path(script_dir):
    """Resolve the project cfg like pygeonet_prepare.py does at import.

    Returns (path, None) on success, (None, reason) when prepare would
    crash or end up without a readable cfg.
    """
    pointer_path = os.path.join(script_dir, 'project_pointer.cfg')
    if os.path.exists(pointer_path):
        parser = configparser.ConfigParser()
        try:
            parser.read(pointer_path)
            return parser.get('CFG Directory', 'project_cfg_pointer'), None
        except configparser.Error as exc:
            return None, 'project_pointer.cfg could not be read: %s' % exc
    # Same glob prepare uses; sorted so the picked cfg is deterministic
    # (prepare itself takes arbitrary glob order).
    matches = sorted(glob.glob(os.path.join(script_dir, 'GeoNet*.cfg')))
    if not matches:
        return None, ('no GeoNet*.cfg found in %s. Run pygeonet_configure.py '
                      'first.' % script_dir)
    return matches[0], None


def read_cfg(cfg_path):
    """Parse the project cfg into the five keys prepare reads.

    Raises SetupError with a readable message when the file is missing,
    unparsable, or missing the [Section] keys.
    """
    parser = configparser.ConfigParser()
    try:
        if not parser.read(cfg_path):
            raise SetupError('cfg file %s does not exist or is not readable'
                             % cfg_path)
        if not parser.has_section(CFG_SECTION):
            raise SetupError('cfg file %s has no [%s] section'
                             % (cfg_path, CFG_SECTION))
        missing = [key for key in CFG_KEYS
                   if not parser.has_option(CFG_SECTION, key)]
        if missing:
            raise SetupError('cfg file %s is missing %s in [%s]'
                             % (cfg_path, ', '.join(missing), CFG_SECTION))
        return {key: parser.get(CFG_SECTION, key) for key in CFG_KEYS}
    except SetupError:
        raise
    except configparser.Error as exc:
        raise SetupError('cfg file %s could not be parsed: %s'
                         % (cfg_path, exc))


def expected_layout(cfg):
    """Paths the pipeline expects, per pygeonet_prepare.py's structure.

    Kinds: 'dir'/'file' fail when missing; 'dir-warn' only warns,
    because pygeonet_prepare.py creates those directories on import.
    """
    home = cfg['geofloodhomedir']
    project = cfg['projectname']
    dem_file = cfg['dem_name'] + '.tif'
    inputs = cfg['Input_dir']
    outputs = cfg['Output_dir']
    return {
        'GeoNet home directory': ('dir', home),
        'Input DEM (%s)' % dem_file: (
            'file', os.path.join(home, inputs, 'GIS', project, dem_file)),
        'Inputs/GIS/%s' % project: (
            'dir-warn', os.path.join(home, inputs, 'GIS', project)),
        'Outputs/GIS/%s' % project: (
            'dir-warn', os.path.join(home, outputs, 'GIS', project)),
        'Outputs/Inundation/%s' % project: (
            'dir-warn', os.path.join(home, outputs, 'Inundation', project)),
    }


def run_checks(cfg):
    """Check every expected path; returns (level, label, detail) tuples."""
    results = []
    for label, (kind, path) in expected_layout(cfg).items():
        if kind.startswith('dir'):
            exists = os.path.isdir(path)
        else:
            exists = os.path.isfile(path)
        if exists:
            results.append((OK, label, path))
        elif kind == 'dir-warn':
            results.append((WARN, label,
                            '%s (missing; pygeonet_prepare.py creates it '
                            'on import)' % path))
        else:
            results.append((FAIL, label, '%s (missing)' % path))
    return results


def probe_imports(module_names=PIPELINE_IMPORTS):
    """Try importing each module; returns (name, imported, detail) tuples.

    Any exception on import counts as unusable for the pipeline, and the
    error text is the diagnostic (e.g. GDAL ABI mismatches surface here
    with their real message instead of inside a stage).
    """
    results = []
    for name in module_names:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # a broken import is exactly what we probe for
            results.append((name, False, '%s: %s' % (type(exc).__name__, exc)))
        else:
            version = getattr(module, '__version__', None)
            results.append((name, True, str(version) if version
                            else 'imported'))
    return results


def main(argv=None, script_dir=None, probe_names=PIPELINE_IMPORTS):
    """Run the checks; returns 0 when nothing failed, 1 otherwise."""
    parser = argparse.ArgumentParser(
        description='Pre-flight checks for the GeoNet/GeoFlood pipeline: '
                    'cfg, home directory, input DEM, project directories.')
    parser.add_argument('--imports', action='store_true',
                        help='also probe the third-party modules the '
                             'pipeline imports and fail if any are missing')
    args = parser.parse_args(argv)
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))

    print('GeoNet setup check')
    cfg_path, reason = resolve_cfg_path(script_dir)
    if cfg_path is None:
        print('FAIL cfg resolution: %s' % reason)
        return 1
    print('using cfg: %s' % cfg_path)
    try:
        cfg = read_cfg(cfg_path)
    except SetupError as exc:
        print('FAIL cfg parse: %s' % exc)
        return 1

    failures = 0
    for level, label, detail in run_checks(cfg):
        print('%s %s: %s' % (level.upper(), label, detail))
        if level == FAIL:
            failures += 1

    if args.imports:
        print('dependency imports:')
        for name, imported, detail in probe_imports(probe_names):
            print('%s %s: %s' % ('OK' if imported else 'FAIL', name, detail))
            if not imported:
                failures += 1

    if failures:
        print('setup check FAILED with %d problem(s)' % failures)
        return 1
    print('setup check passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
