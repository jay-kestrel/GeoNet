"""Regression tests: stage-5 Numba kernel determinism (GeoNet todo_vj7M7zN1).

``Fast_Marching_Start_Point_Identification`` historically appended to two shared
Python lists inside ``prange``: outputs came out in thread-schedule order (not
label order), appends were lost, and the heap corrupted (SIGSEGV / SIGABRT —
reproduced 26/26 fresh parallel processes on one geometry sweep).

The kernel now performs disjoint per-label writes into a boolean mask, and
``fmm_list_creation`` gathers start points in label order (the deterministic
serial reduction). These tests pin that contract:

- repeated parallel runs are byte-identical,
- outputs match a pure-numpy label-order reference exactly,
- both selection gates (percent of DEM area, minimum basin elements) are
  exercised, including cases where each gate alone excludes a basin,
- degenerate array-edge shapes (1-row, 1-column, 1-pixel, zero-outlet) run.

Tests run under the default numba thread count on purpose — the parallel path
is what must be deterministic. Do not set NUMBA_NUM_THREADS=1 here to make
them pass.
"""
import numpy as np
import pytest

import pygeonet_fast_marching as fm

BASIN_ELEMENTS = 2   # Parameters.numBasinsElements
THRESHOLD = 0.1      # defaults.thresholdPercentAreaForDelineation


def build_case(ny, nx, rects, dtype=np.float64):
    """Basin raster from explicit (r0, r1, c0, c1) rectangles, one outlet each.

    Basin ids are 1..B in the order given (matching GRASS basin numbering and
    main()'s argwhere-derived outlet list). Remaining pixels stay 0.
    """
    basin = np.zeros((ny, nx), dtype=dtype)
    outlets = []
    for i, (r0, r1, c0, c1) in enumerate(rects):
        basin[r0:r1, c0:c1] = i + 1
        outlets.append((r0, c0))
    splist = np.transpose(np.array(outlets, dtype=np.int64)).reshape(2, -1)
    return basin, splist


def reference_keep(basin, splist, basin_elements=BASIN_ELEMENTS,
                   threshold=THRESHOLD):
    """Pure-numpy ground truth: per-label count, label-order keep mask."""
    iter_total = splist.shape[1]
    keep = np.zeros(iter_total, dtype=bool)
    nDempixels = float(basin.size)
    for label in range(iter_total):
        n = np.count_nonzero(basin == label + 1)
        if (n * 100.00001 / nDempixels > threshold) and (n > basin_elements):
            keep[label] = True
    return keep


def run_kernel(basin, splist):
    iter_total = np.arange(0, splist.shape[1]).size
    return fm.Fast_Marching_Start_Point_Identification(
        None, basin, splist, float(basin.size), BASIN_ELEMENTS, THRESHOLD,
        iter_total)


@pytest.fixture(scope="session")
def grid_case():
    """36 basins of 16x16 on a 96x96 DEM — every basin qualifies."""
    rects = [(r, r + 16, c, c + 16)
             for r in range(0, 96, 16) for c in range(0, 96, 16)]
    return build_case(96, 96, rects)


def test_repeated_runs_identical(grid_case):
    basin, splist = grid_case
    first = run_kernel(basin, splist)
    first_fmm = fm.fmm_list_creation(splist, first)
    for _ in range(12):
        keep = run_kernel(basin, splist)
        fmm = fm.fmm_list_creation(splist, keep)
        assert np.array_equal(keep, first)
        assert fmm.tobytes() == first_fmm.tobytes()
        assert fmm.shape == first_fmm.shape


def test_matches_pure_numpy_reference(grid_case):
    basin, splist = grid_case
    keep = run_kernel(basin, splist)
    ref = reference_keep(basin, splist)
    assert keep.dtype == np.bool_
    assert np.array_equal(keep, ref)
    assert keep.all()  # every 16x16 basin is 2.78% of the DEM -> all qualify
    fmm = fm.fmm_list_creation(splist, keep)
    assert np.array_equal(fmm, splist[:, ref])  # exact values, label order


def test_mixed_qualification_gates():
    """Isolate each exclusion gate on a 20x20 DEM.

    Percent gate passes for any n >= 1 (0.25% minimum > 0.1%), so the 1-px and
    2-px basins below are excluded by the minimum-elements gate alone, while
    the 3-px strips qualify (0.75%, n=3 > 2). Basins 6+ (1-px filler) are all
    excluded.
    """
    rects = [
        (0, 3, 0, 3),        # 1: 9 px  -> qualify
        (0, 3, 3, 6),        # 2: 9 px  -> qualify
        (0, 3, 6, 9),        # 3: 9 px  -> qualify
        (3, 4, 0, 3),        # 4: 3 px strip -> qualify
        (3, 4, 3, 6),        # 5: 3 px strip -> qualify
        (4, 5, 0, 2),        # 6: 2 px  -> elements gate only
        (4, 5, 3, 5),        # 7: 2 px  -> elements gate only
        (5, 6, 0, 1),        # 8: 1 px  -> elements gate only
        (5, 6, 1, 2),        # 9: 1 px  -> elements gate only
    ]
    basin, splist = build_case(20, 20, rects)
    keep = run_kernel(basin, splist)
    ref = reference_keep(basin, splist)
    assert np.array_equal(keep, ref)
    expected = np.zeros(splist.shape[1], dtype=bool)
    expected[:5] = True
    assert keep.shape == (splist.shape[1],)
    assert np.array_equal(keep, expected)
    fmm = fm.fmm_list_creation(splist, keep)
    assert fmm.shape == (2, 5)
    # label order: the five qualifying outlets, rows then cols
    assert np.array_equal(fmm[0], [0, 0, 0, 3, 3])
    assert np.array_equal(fmm[1], [0, 3, 6, 0, 3])


@pytest.mark.parametrize("ny,nx", [(1, 32), (32, 1), (2, 2), (1, 1)])
def test_array_edge_shapes(ny, nx):
    """Degenerate raster edges: 1-row, 1-column, 2x2, and 1-pixel DEMs."""
    rects = [(0, 1, 0, 3)] if nx >= 3 else [(0, 1, 0, 1)]
    basin, splist = build_case(ny, nx, rects)
    keep = run_kernel(basin, splist)
    ref = reference_keep(basin, splist)
    assert np.array_equal(keep, ref)
    fmm = fm.fmm_list_creation(splist, keep)
    assert fmm.shape[0] == 2
    assert fmm.shape[1] == int(keep.sum())


def test_zero_outlet_list():
    """No outlets at all: empty mask and an empty (2, 0) FMM list."""
    basin = np.zeros((8, 8), dtype=np.float64)
    splist = np.zeros((2, 0), dtype=np.int64)
    keep = run_kernel(basin, splist)
    assert keep.shape == (0,)
    fmm = fm.fmm_list_creation(splist, keep)
    assert fmm.shape == (2, 0)


@pytest.mark.parametrize("dtype", [np.float64, np.int16, np.int32])
def test_basin_raster_dtypes(dtype):
    """Basin rasters arrive as Int16 GeoTIFFs (or float64 after read) — the
    kernel must treat both identically."""
    rects = [(0, 8, 0, 8), (0, 8, 8, 16), (8, 16, 0, 8), (8, 16, 8, 16)]
    basin, splist = build_case(16, 16, rects, dtype=dtype)
    keep = run_kernel(basin, splist)
    assert np.array_equal(keep, reference_keep(basin, splist))
    assert keep.all()
