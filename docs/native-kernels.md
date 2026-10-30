<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Native kernel scaffold

This builds `geonet_kernels`; it does **not** accelerate GeoNet yet. Both native
functions deliberately contain `unimplemented!()`. No pipeline module imports
the new dispatcher, so installing the wheel leaves existing scripts unchanged.
Do not route production calls into the scaffold.

## Build and test

Run from the repository root with Rust installed and the desired Python active:

```sh
python -m pip install maturin==1.16.0 pytest==7.4.4 tifffile==2021.11.2
maturin build --release --interpreter "$(command -v python)" --out wheelhouse
python -m pip install --no-deps wheelhouse/*.whl
python -c 'import geonet_kernels; print(geonet_kernels.__file__)'
python -m pytest tests/test_kernel_dispatch.py tests/test_native_scaffold.py tests/test_golden_helpers.py
cargo fmt --manifest-path rust/geonet-kernels/Cargo.toml -- --check
cargo clippy --locked --manifest-path rust/geonet-kernels/Cargo.toml --all-targets -- -D warnings
cargo test --locked --manifest-path rust/geonet-kernels/Cargo.toml
maturin sdist --out dist
```

Install NumPy in the active environment before runtime tests; use `numpy==1.19.1`
for the Python 3.7.7 benchmark environment. The wheel contains only the native
module and license, not the flat Python pipeline. Use the dispatcher from a
GeoNet checkout with its normal Python dependencies installed.

Maturin enables `pyo3/extension-module` only for extension builds. `cargo test`
needs an embeddable Python with development/link libraries. For a relocated conda
environment, set `PYO3_PYTHON=$CONDA_PREFIX/bin/python`,
`PYTHONHOME=$CONDA_PREFIX`, and, on Linux,
`LD_LIBRARY_PATH=$CONDA_PREFIX/lib` for the cargo test command.

The existing suite needs scikit-fmm 2021.7.8 or newer for its negative-discriminant
regression test. The **frozen benchmark instead uses scikit-fmm 2019.1.30**. Keep
these environments separate; upgrading the benchmark solver invalidates that
controlled comparison. GRASS-dependent tests skip when GRASS is unavailable.

## Q1: Python 3.7.7 support verified

On 2026-10-30, the exact CPython 3.7.7 interpreter (`hcff3b4d_5` from the archived
conda lock), NumPy 1.19.1, PyO3 **0.28.3**, rust-numpy **0.28.0**, and maturin
**1.16.0** built and installed a `cp37-abi3` Linux wheel. Cargo registration tests
and 27 dispatch/native tests passed. The installed module exposed these actual
Python signatures:

```python
identify_start_points(outlet_array, basinIndexArray, fastMarchingStartPointList,
                      nDempixels, basin_elements, threshold, iter_total)
find_channel_heads(skeletonFromFlowAndCurvatureArray, geodesicDistanceArray)
```

The pins are based on direct source/package metadata inspection, then a real
build—not the spec's tentative 0.20.x assumption. PyO3 0.28.3 declares a Python
3.7 minimum; 0.29.0 raises it to 3.8. Maturin 1.16.0 declares `Requires-Python:
>=3.7`. Sources fetched on 2026-10-30:

- [PyO3 0.28.3 build configuration](https://raw.githubusercontent.com/PyO3/pyo3/v0.28.3/pyo3-build-config/src/impl_.rs)
- [PyO3 0.29.0 build configuration](https://raw.githubusercontent.com/PyO3/pyo3/v0.29.0/pyo3-build-config/src/impl_.rs)
- [Maturin package metadata](https://pypi.org/pypi/maturin/json)
- [rust-numpy 0.28.0 requirements](https://raw.githubusercontent.com/PyO3/rust-numpy/v0.28.0/README.md)

The crate uses the Python 3.7 stable ABI. CI builds wheels on Linux x86_64,
macOS Intel, and Windows AMD64 with cibuildwheel 2.23.3, whose build matrix still
contains CPython 3.7. A separate job builds on **exactly 3.7.7**; wheel-platform
jobs use their available 3.7 patch release. These build checks do not certify
scientific parity or full GeoNet compatibility on every platform.

## Run-only CI is intentional

Both workflows contain only shell steps. On 2026-10-30, GitHub rejected even
first-party actions with the policy clause “all actions must be from a repository
owned by jay-kestrel” ([test job](https://github.com/jay-kestrel/GeoNet/actions/runs/41112648418/job/124258693275),
[Windows wheel job](https://github.com/jay-kestrel/GeoNet/actions/runs/41112648627/job/124258694491)).
Do not replace these steps with third-party or first-party Actions without an
administrator-authorized policy change.

Shell checkout fetches the exact event SHA from this public repository, without
persisting credentials. Rust uses the runner's `rustup`; cibuildwheel 2.23.3 runs
as a Python CLI and acquires its own build interpreters. The exact-3.7.7 job uses
the locally verified Anaconda-channel recipe via a checksum-pinned standalone
micromamba 2.10.0-0 binary, not the micromamba Action. The binary URL and SHA-256
were retrieved from its GitHub release metadata on 2026-10-30. The separate full
pytest job uses Ubuntu 22.04's runner Python in an isolated virtual environment;
it is not the frozen benchmark environment.

Each matrix job audits its own platform wheel's native module, cp37 ABI, and GPL
license, then prints its path and SHA-256. The exact-interpreter job also logs the
sdist hash; command failures retain their nonzero job status. This preserves
build/test coverage but gives up downloadable workflow artifacts and a combined
cross-platform artifact audit. All matrix jobs must pass. An administrator
allowlist could restore conventional Actions and artifact delivery later; that
policy preference is outside this PR.

## Dispatch contract

`pygeonet_kernels.py` reads `GEONET_KERNELS` on every call. `python` forwards to
the original stage function; `auto` prefers the native module and falls back
only when that module is absent; `rust` raises if the module is absent. Invalid
values, missing native dependencies, broken binaries, and execution errors
propagate. There is no silent retry through Python after a native failure.

Array arguments are accepted as NumPy arrays without choosing a single dtype in
this scaffold: archived basin rasters are integer-valued, while existing Python
tests also use floating arrays. The implementation PRs own dtype/shape validation
and byte equivalence. Stage 5 returns the boolean qualification mask; stage 6
returns `(xx, yy)`. The Python stage-6 function also plots and writes files;
those side effects must stay Python-owned when the scan is integrated.

## Frozen golden evidence

Use only archived `e7e1bdb6429e58d57112cb6bffa2f4de2626f61b` outputs. The
`cbb071f` outputs contain the lost-basin regression and are never a valid oracle.

Download `fl_NGrEVwuj` (`geonet-e7e1bdb-original-upstream-evidence.tar.gz`) from
the approved benchmark evidence. Its verified SHA-256 is:

```text
72b1a94825dcc8e59ad7628acfe0e74ff30d8ec380aa3028e843b246cc888446
```

Verify that checksum before extracting into a read-only oracle directory. Set
`GEONET_GOLDEN_ROOT` to the extracted root containing `data/GeoInputs` and
`data/GeoOutputs`, then run:

```sh
GEONET_GOLDEN_ROOT=/absolute/path/to/extracted-bundle \
  python -m pytest tests/test_golden_helpers.py -v
```

`tests/golden_manifest.json` pins all three input hashes and every archived
raster hash, derived from `frozen-inputs.json` and `canonical-output-audit.json`.
The archive, all three inputs, and all 51 output files were independently hashed
and matched on 2026-10-30 before the manifest was created. No expected output is
regenerated by a test.

`golden.load_input` and `golden.load_output` verify a file before loading it.
`assert_array_bytes_equal` checks shape, dtype, and C-order pixel bytes, including
NaN payloads and signed zero. `assert_raster_matches_golden` checks exact TIFF
file bytes, including metadata. Fixture-integrity tests are **not** a claim that
unfinished Rust kernels pass A1. No environment variable means an explicit skip
with the missing-bundle reason; a configured but missing/corrupted file fails.

## Licensing and boundaries

New hand-authored files use `SPDX-License-Identifier: GPL-3.0-or-later` (JSON uses
its `license` field). The upstream `License.txt` is unchanged and ships in the
wheel. Cargo.lock is generated dependency metadata. PyO3, ndarray, and Rayon
use MIT/Apache-2.0 alternatives; rust-numpy uses BSD-2-Clause. The resolved
transitive graph was reviewed on 2026-10-30; CI rejects license expressions
outside `tools/check_native_licenses.py`'s explicit permissive allowlist.

GRASS, scikit-fmm, diffusion, and all raster/vector I/O remain outside the native
crate. No pipeline call-site changes belong in this scaffold. Backend-only work
has no rendered UI or interactive visual state, so browser screenshots are not
applicable evidence.
