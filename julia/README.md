# GeoNetPorts — Julia ports of GeoNet's core numeric functionality

A self-contained Julia subproject that ports selected numeric kernels from
the Python GeoNet pipeline. It reads plain arrays / `.npy` fixtures and never
imports the Python modules. One submodule per Python stage, mirroring the
`pygeonet_*` module names.

## What is ported

### `FastMarching` — hand-rolled eikonal (fast marching) solver

`src/FastMarching.jl` implements the scikit-fmm travel-time solve that the
GeoNet pipeline delegates to (`pygeonet_fast_marching.py` →
`skfmm.travel_time`). It is a close transcription of skfmm's C++ marcher
(`base_marcher.{h,cpp}`, `distance_marcher.cpp`, `travel_time_marcher.{h,cpp}`,
`heap.{h,cpp}`), preserving:

- zero-contour initialization: `phi == 0` cells frozen at 0; straddling cells
  get the per-dimension interpolated crossing distance combined as
  `sqrt(1/Σ 1/dᵢ²)`, converted to time via `|distance / speed|`;
- the zero-speed mask rule (`speed < double-epsilon → wall`) and the
  maxDouble → NaN mapping on output (skfmm's masked cells);
- order-1 (first-order) and order-2 (second-order, the skfmm default)
  point updates, including the negative-discriminant retry that drops one
  dimension at a time and keeps the smallest reduced solution;
- the exact heap mechanics (indexed binary min-heap with back pointers, strict
  `<` comparisons, skfmm's set/pop sift order) and the batch freezing of
  exact-tie minima, so tie resolution and therefore marching order match;
- scalar or per-axis grid spacing (`dx::Real` or `dx::NTuple{2,Real}`).

```julia
using GeoNetPorts
t = travel_time(phi, speed, 0.01; order = 2)          # skfmm.travel_time analog
t = travel_time(phi, speed, (2.0, 1.0); order = 2)    # anisotropic spacing
```

Errors mirror skfmm's: `ArgumentError` for shape/spacing/order problems,
`FastMarchingError("the array phi contains no zero contour …")` for the
"no zero contour" case, and `FastMarchingError("Unreachable voxel")` for
skfmm's unreachable-voxel runtime error.

## Equivalence gate

`test/runtests.jl` replays the committed skfmm goldens and asserts
**≤ 1e-6 relative agreement** on every unmasked cell plus identical
masked-cell sets. Measured on this branch: **max relative error 0.0
(bit-exact) on all seven cases**, including the 128×128 two-lobe basin with
an interior zero-speed wall (11,888 masked cells) where skfmm's
negative-discriminant and tie-freezing paths both execute. 52 tests pass.

| Case | Terrain | Order | Masked cells |
|---|---|---|---|
| `point_source_constant` | 64² point source, constant speed | 2 | 0 |
| `smooth_hills` | 128² smooth speed field | 2 | 0 |
| `rough_basin_masked` | 128² two-lobe basin + interior zero-speed wall | 2 | 11,888 |
| `edge_start_rough` | 96² rough speed, start near a corner | 2 | 0 |
| `anisotropic_dx` | 64² `dx = (2.0, 1.0)` | 2 | 0 |
| `row_1d` | 1×8 single-row grid | 2 | 0 |
| `order1_smooth` | 64² first-order updates | 1 | 0 |

### Regenerating the goldens

The goldens are committed (`.npy` files pinned by `test/fixtures/manifest.sha256`,
same doctrine as the COBOL track's benchmark fixtures). Regenerate with the
reference stack only — numpy 2.3.5 / scikit-fmm 2025.6.23 in the project
sandbox — and review the diff; the digests must reproduce bit-for-bit:

```bash
cd julia/test/fixtures && python3 generate_fmm_goldens.py
```

## Running

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'   # tests
julia --project=. bench/fmm_bench.jl                              # benchmark
```

Julia ≥ 1.11. No build step; the solver is pure Julia with no runtime
dependencies.

## Benchmark notes (informational, same bytes, warm, best-of-5)

128×128 `rough_basin_masked`: skfmm 1.2 ms vs Julia 1.5 ms.
128×128 `smooth_hills`: skfmm 4.1 ms vs Julia 5.1 ms.
The unoptimized, equivalence-first Julia port is warm-competitive with the
mature C extension at pipeline-relevant sizes; the bench ladder in
`bench/fmm_bench.jl` documents scaling (≈14 ms at 256², ≈90 ms at 512²,
≈400–500 ms at 1024² on the project sandbox). No speed claim is made or
implied; comparisons are same-revision, same-machine observations.

## CI

`.github/workflows/julia.yml` runs the test suite on every push and pull
request. Per repository policy it uses no external actions: it checks out the
event commit inline and installs a pinned Julia 1.11.1 from the official CDN
in a `run:` step.

## Scope boundary

The fast-marching solve is ported; GRASS flow accumulation / basin indexing
(stage 3) stays external to this subproject, per the build spec.
