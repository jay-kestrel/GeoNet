#! /usr/bin/env python
"""Environment report for GeoNet bug reports.

    python pygeonet_envinfo.py

Prints the Python version, platform, git commit (when available), and
the installed version of every third-party dependency the pipeline
imports. Modules that cannot be imported are reported as 'not
installed' instead of failing: the report must always succeed so it can
be pasted straight into an issue.

GeoNet has no version string anywhere, and its recent history is a
series of version-compatibility fixes, so bug reports need this
context.

Stdlib-only at module level; heavy dependencies are only touched inside
guarded probes, so importing this module has no side effects.
"""
import importlib
import os
import platform
import subprocess
import sys

# Third-party modules the pipeline imports at module level. GRASS is
# probed as grass.script, the import pygeonet_grass_py3.py uses.
DEPENDENCIES = (
    'numpy', 'scipy', 'matplotlib', 'osgeo.gdal', 'rasterio',
    'numba', 'skfmm', 'psutil', 'dask', 'statsmodels', 'grass.script',
)


def git_commit(repo_dir):
    """Short HEAD commit of the repo containing repo_dir, or None."""
    try:
        result = subprocess.run(
            ('git', 'rev-parse', '--short', 'HEAD'),
            cwd=repo_dir, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.decode('ascii', 'replace').strip() or None


def dependency_version(name):
    """Installed version of a dependency, or 'not installed'.

    Any exception on import counts as not usable for the pipeline; this
    report must never crash on a missing or broken dependency.
    """
    try:
        module = importlib.import_module(name)
    except Exception:  # a missing/broken import is exactly what we report
        return 'not installed'
    version = getattr(module, '__version__', None)
    return str(version) if version else 'installed (unknown version)'


def collect_env_info(repo_dir=None, dependencies=DEPENDENCIES):
    """Gather the environment details a bug report should include."""
    if repo_dir is None:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'git_commit': git_commit(repo_dir),
        'dependencies': {name: dependency_version(name)
                         for name in dependencies},
    }


def format_env_info(info):
    """Render collect_env_info output as pasteable text."""
    lines = [
        'GeoNet environment information',
        'Python: %s' % info['python'],
        'Platform: %s' % info['platform'],
        'Git commit: %s' % (info['git_commit']
                            or 'unknown (not a git repository)'),
        'Dependencies:',
    ]
    for name, version in info['dependencies'].items():
        lines.append('  %s: %s' % (name, version))
    return '\n'.join(lines)


def main(repo_dir=None, dependencies=None):
    """Print the environment report; always returns 0."""
    print(format_env_info(collect_env_info(
        repo_dir=repo_dir, dependencies=dependencies or DEPENDENCIES)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
