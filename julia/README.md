# GeoNetPorts — Julia port of GeoNet's core numeric kernels

Self-contained Julia subproject porting the in-repo numeric core of the
Python GeoNet pipeline. It never imports the Python package; equivalence is
established against committed goldens produced by the original kernels.

SPDX-License-Identifier: GPL-3.0-or-later (matches the repository license).

## Scope

Ported here (stage references are the Python pipeline stages):

| Julia module | Python source | Gate |
|---|---|---|
| `NonlinearFilter` — `simple_gaussian_smoothing`, `anisodiff` (both conductance options), `lambda_nonlinear_filter` | `pygeonet_nonlinear_filter.py` | bit-exact; smoothing convolution <=1e-6 relative |
| `SlopeCurvature` — `compute_dem_slope`, `compute_dem_curvature` (geometric + laplacian) and the curvature mean/std | `pygeonet_slope_curvature.py` | bit-exact |
| `StartPoints` — `fast_marching_start_point_identification` (Numba `prange` kernel), `fmm_list_creation` | `pygeonet_fast_marching.py` | bit-exact, oracle-checked, thread-schedule independent |
| `NumPyCompat` — numpy pairwise summation, `np.gradient` scheme, `nanmean`/`nanstd`, scipy `mquantiles` (alphap=betap=0.4) | numpy/scipy internals | bit-exact on committed fixtures |

Consciously excluded (unchanged from the port spec): the scikit-fmm
eikonal solve, GRASS flow accumulation / basins / outlets, GeoTIFF and
shapefile I/O, plotting and config plumbing. The module list under
`src/GeoNetPorts.jl` is the extension point — the standalone fast-marching
solver lands beside these as another include without touching existing
kernels.

## Layout

```
julia/
├── Project.toml            # package manifest; test deps are extras (NPZ, Test, SHA)
├── src/                    # GeoNetPorts + one module per Python stage
├── test/runtests.jl        # equivalence suite (the gates above)
├── test/goldens/           # committed .npy fixtures + manifest.json (sha256)
└── tools/generate_goldens.py
```

## Running

```
julia --project=julia -e 'using Pkg; Pkg.test()'
```

CI (`.github/workflows/julia.yml`) runs the same suite with a pinned
Julia release downloaded in-script — the workflow uses no external
actions, per repository policy.

## Goldens and equivalence doctrine

`tools/generate_goldens.py` runs the original Python kernels on seeded
synthetic inputs (96x96 fractal DEM with injected NaN holes and `-9998`
flag patches; three hand-built basin geometries for the start-point
kernel; 67 size-1..1000 scipy-quantile samples) and writes every
input/output as `.npy` plus a sha256 `manifest.json`. It needs a Python
environment with numpy/scipy/numba (import-time stubs stand in for
rasterio/GRASS/skfmm so only the kernels under test run real code); the
reference stack is recorded in the manifest. Regenerate with
`python3 julia/tools/generate_goldens.py` — seeds and stack pin the
output; CI fails if a committed fixture stops matching its hash.

Gates and their rationale:

- **Bit-exact (raw bytes, NaN placement included)** for every elementwise
  kernel. This holds because the port reproduces numpy's exact float32
  arithmetic: weak-scalar promotion keeps scalars in the array dtype, and
  reductions use numpy's pairwise summation order (zero seed, eight
  accumulators combined two-by-two up to 128, divide and conquer beyond —
  `NumPyCompat.numpy_pairwise_sum`; numpy's internally `-0.0`-seeded
  variant sits behind an add-identity pre-add, so the zero seed matches
  the observable result, including all-negative-zero inputs).
- **<=1e-6 relative** for `simple_gaussian_smoothing` (a float64
  convolution): the port accumulates the kernel in row-major order while
  scipy's C loop may order differently; both float64 sums agree to
  ~1e-16 on the fixtures (measured 6.1e-16), far under the gate.
- **Reduction order matters beyond the algorithm**: element extraction
  feeding `nanmean`/`nanstd` must iterate row-major (numpy ravel order) —
  a column-major pass reorders the pairwise tree and drifts ulps (this
  was a real port bug, fixed and pinned by the curvature-stats gate).
- The oracle for the start-point kernel is the pure-NumPy reference from
  `tests/test_stage5_determinism.py` (`reference_keep`), ported in the
  test suite; kernel, oracle, and the real Numba kernel's golden output
  must all agree, with repeated runs identical regardless of thread
  scheduling (disjoint per-label writes, serial label-order reduction).

Known semantic quirks preserved on purpose (bug-for-bug): the
`100.00001` area factor and `n > 2` basin-element gate; the nan-mean
padding recipe whose corner cells inherit the horizontal means; the
`deltaS`/`deltaE` last row/column staying zero across anisodiff
iterations; label lookups keyed on the python 0-based `label + 1`.
