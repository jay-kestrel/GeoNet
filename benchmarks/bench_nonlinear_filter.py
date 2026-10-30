"""Wall-clock benchmark for the Perona-Malik nonlinear filter.

Times pygeonet_nonlinear_filter.anisodiff on the committed synthetic DEM
fixture with each available backend:

- ``python``: the NumPy reference path (default routing).
- ``cobol``:  the COBOL kernel behind PYGEONET_COBOL=1.

Rows are backend-tagged (spec AC3). The COBOL kernel is single-threaded and
GnuCOBOL statement-interpreted; per the port spec there is NO performance
gate - timings are feasibility measurements only.

Output: one JSON object with a ``manifest`` (environment provenance) and
``rows`` (one row per backend per repeat aggregate), printed to stdout and
written to ``--output`` when given. Run from any directory:

    python benchmarks/bench_nonlinear_filter.py --output /tmp/bench.json
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import types
import unittest.mock as mock

import numpy as np

# Stub the heavy geospatial import chain (same pattern as the tests) so the
# benchmark needs only numpy + scipy + the built kernel library.
def _install_stub_modules():
    def plain_module(name):
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module

    osgeo = plain_module('osgeo')
    for submodule in ('gdal', 'osr', 'ogr'):
        sub = types.ModuleType('osgeo.' + submodule)
        setattr(osgeo, submodule, sub)
        sys.modules['osgeo.' + submodule] = sub

    rasterio = plain_module('rasterio')
    rasterio_crs = plain_module('rasterio.crs')
    rasterio_crs.CRS = type('CRS', (), {})
    rasterio.crs = rasterio_crs

    matplotlib = plain_module('matplotlib')
    matplotlib.pyplot = plain_module('matplotlib.pyplot')
    matplotlib.cm = plain_module('matplotlib.cm')
    matplotlib.cm.__getattr__ = lambda name: name
    matplotlib.use = lambda *a, **k: None

    plain_module('pygeonet_prepare')
    plain_module('pygeonet_defaults')


_install_stub_modules()

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pygeonet_nonlinear_filter as nlf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIXTURE = os.path.join(
    HERE, 'fixtures', 'synthetic_dem_128x128.npy')


def sha256_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def gnucobol_version():
    """Parse `cobc --version`; None when the toolchain is absent."""
    try:
        out = subprocess.run(['cobc', '--version'], capture_output=True,
                             text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.stdout.splitlines():
        if 'GnuCOBOL' in line:
            parts = line.replace(')', ' ').split()
            for part in parts:
                if part[0:1].isdigit():
                    return part
    return None


def git_commit():
    try:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
            check=True, cwd=os.path.dirname(HERE)).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def collect_manifest(fixture_path):
    manifest = {
        'bench': 'bench_nonlinear_filter.py',
        'git_commit': git_commit(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'fixture': os.path.basename(fixture_path),
        'fixture_sha256': sha256_of(fixture_path),
        'gnucobol_version': gnucobol_version(),
        'cobol_single_threaded': True,
        'note': 'COBOL rows are feasibility/parity measurements; '
                'no performance gate (spec art_QEojt4gd decision 10).',
    }
    return manifest


def time_backend(backend, img, niter, kappa, gamma, step, option,
                 warmup, repeats):
    """Return per-repeat wall-clock seconds for one backend."""
    with mock.patch.dict(os.environ, {'PYGEONET_COBOL': '1'} if backend ==
                         'cobol' else {}):
        for _ in range(warmup):
            nlf.anisodiff(img, niter, kappa, gamma, step, option)
        timings = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            nlf.anisodiff(img, niter, kappa, gamma, step, option)
            timings.append(time.perf_counter() - t0)
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['python', 'cobol', 'both'],
                        default='both')
    parser.add_argument('--repeats', type=int, default=3,
                        help='timed repeats per backend (default 3)')
    parser.add_argument('--warmup', type=int, default=1,
                        help='untimed warmup calls per backend (default 1)')
    parser.add_argument('--niter', type=int, default=10,
                        help='diffusion iterations (default 10)')
    parser.add_argument('--kappa', type=float, default=2.5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--step1', type=float, default=1.0)
    parser.add_argument('--step2', type=float, default=1.0)
    parser.add_argument('--option', type=int, default=2, choices=[1, 2],
                        help='Perona-Malik conductance option (default 2)')
    parser.add_argument('--fixture', default=DEFAULT_FIXTURE)
    parser.add_argument('--output', default=None,
                        help='also write the JSON document to this path')
    args = parser.parse_args()

    img = np.load(args.fixture)
    step = (args.step1, args.step2)
    backends = ['python', 'cobol'] if args.backend == 'both' else [args.backend]

    rows = []
    reference_result = None
    for backend in backends:
        timings = time_backend(backend, img, args.niter, args.kappa,
                               args.gamma, step, args.option,
                               args.warmup, args.repeats)
        result = nlf.anisodiff(img, args.niter, args.kappa, args.gamma,
                               step, args.option)
        if backend == 'python':
            reference_result = result
            agreement = None
        elif reference_result is not None:
            finite = ~np.isnan(reference_result)
            agreement = float(np.max(np.abs(
                result[finite] - reference_result[finite])))
        else:
            agreement = None
        row = {
            'kernel': 'nonlinear_filter',
            'backend': backend,
            'implementation': 'cobol-gnucobol' if backend == 'cobol'
                             else 'numpy',
            'iterations': args.niter,
            'option': args.option,
            'kappa': args.kappa,
            'gamma': args.gamma,
            'step': [args.step1, args.step2],
            'shape': list(img.shape),
            'warmup_calls': args.warmup,
            'repeat_timings_s': [float(t) for t in timings],
            'min_s': float(min(timings)),
            'median_s': float(np.median(timings)),
            'max_s': float(max(timings)),
            'max_abs_dev_vs_python': agreement,
        }
        rows.append(row)

    document = {'manifest': collect_manifest(args.fixture), 'rows': rows}
    text = json.dumps(document, indent=2)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(text)
        print(f'wrote {args.output}')
    print(text)


if __name__ == '__main__':
    main()
