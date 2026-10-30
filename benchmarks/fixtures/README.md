# Benchmark fixtures

Committed deterministic workloads for the GeoNet benchmark scripts.

## `synthetic_dem_128x128.npy`

128x128 float32 DEM-like raster (two ridges, a planar gradient, seeded
noise). Used by `bench_nonlinear_filter.py` (python vs cobol rows) and by
the COBOL parity tests.

- sha256: `e56c7d5387a8b06265ec121fca8e3dbd31a0a5e46b511725c679e363d0a692b4`
- Regenerate with `python3 make_synthetic_dem.py` — the parity test asserts
  the committed digest, so regeneration drift fails CI instead of silently
  moving the benchmark workload.
