# GeoNet COBOL kernels

GnuCOBOL port of the Perona-Malik nonlinear filter (`anisodiff` in
`pygeonet_nonlinear_filter.py`) - part of the workspace COBOL port
program (spec art_QEojt4gd, deliverable D5). The Python module stays the
behavioral reference; the COBOL kernel is a pure compute backend selected
at runtime with `PYGEONET_COBOL=1`.

## Layout

- `nonlinear_filter.cob` - the kernel: per-iteration delta /
  conductance / flux-divergence passes with operation order matching the
  frozen NumPy oracle statement by statement (COMP-1 = float32
  arithmetic, per-op rounding matched to the NumPy chain).
- `shim.c` - ctypes entry point (`ge_anisodiff_cobol`): one-time
  `cob_init()`, scratch allocation for the flux/flag tables, forwards to
  the generated COBOL program. Direct ctypes into a module without
  `cob_init` is not safe - the shim is load-bearing.
- `Makefile` - `cobc -C` translates COBOL to C, then gcc links one
  shared object: `../lib/pygeonet_cobol.so`.

## Build

```sh
apt-get install gnucobol3   # GnuCOBOL 3.2 (ubuntu-latest / dev sandbox)
make -C cobol lib
```

## Use

```python
import pygeonet_nonlinear_filter as nlf
nlf.anisodiff(dem, niter=15, kappa=2.5, gamma=0.1)   # NumPy (default)

# with PYGEONET_COBOL=1 in the environment:
nlf.anisodiff(dem, niter=15, kappa=2.5, gamma=0.1)   # COBOL kernel
```

## Contract notes

- **No silent fallback (C2):** with `PYGEONET_COBOL=1` a missing or
  unloadable kernel raises `FileNotFoundError`; nothing quietly reverts
  to NumPy. Default routing (variable unset) is byte-identical to the
  pre-port implementation (verified in tests).
- **NaN handling:** GnuCOBOL `COMPUTE` does not propagate NaN operands
  (they coerce toward zero), so the kernel tracks NaN with explicit
  per-cell flag tables and synthesizes the IEEE quiet NaN (0x7FC00000)
  only through a REDEFINES alias. Values at flagged positions are
  placeholders and never enter arithmetic. NaN outputs match the NumPy
  reference position-for-position (tests assert mask equality).
- **exp underflow clamp:** GnuCOBOL's `FUNCTION EXP` evaluates
  huge-magnitude arguments in arbitrary-precision decimal and aborts on
  the allocation. float32 `exp(-x)` is 0 for `x >= 104`, so the kernel
  clamps there - identical to the NumPy underflow-to-zero behavior.
- **kappa=0 divergence:** the NumPy reference emits inf with runtime
  warnings; the COBOL backend rejects it with `ValueError` instead.
- **Image size limit:** the ODO tables cap at 2^26 = 67,108,864 pixels
  (8192x8192; GnuCOBOL's per-table byte size overflows past that).
  Larger inputs raise `ValueError` from the adapter.
- **Determinism:** the kernel is stateless, single-threaded, and draws
  no randomness (RNG stays at the Python layer, C5). Repeated calls are
  bit-identical.
- **x86-64 assumption:** the quiet-NaN alias and bit-level fixtures were
  validated on x86-64 Linux; other architectures should re-run the
  parity suite before relying on bit-exactness (tolerance parity holds).

## Verification

`tests/test_nonlinear_filter_cobol.py` gates parity at rtol=1e-6,
atol=2e-6 against the frozen oracle (tests/reference_nonlinear_filter.py,
pinned at 5e2e879) across degenerate shapes, both conductance options,
zero/one/many iterations, layout variants, and NaN patterns. On this
toolchain the kernel is bit-exact against NumPy on every case.

`benchmarks/bench_nonlinear_filter.py` emits backend-tagged timing rows
(python vs cobol) with environment provenance. There is no performance
gate: COBOL rows are feasibility measurements.
