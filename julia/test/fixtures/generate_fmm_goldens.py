"""Regenerate the committed skfmm travel-time goldens (deterministic).

Each case directory under this one holds the inputs (phi.npy, speed.npy,
params.txt) and the scikit-fmm reference output (travel_time.npy, mask.npy)
for one terrain, generated in the calling shape the GeoNet pipeline uses
(phi: 0 outside the basin, 1 inside, -1 at the start point; speed: positive
in the solved region). The Julia equivalence test replays these files through
GeoNetPorts.FastMarching.travel_time and asserts the travel-time fields agree
within 1e-6 relative.

Regenerating must reproduce the committed .npy files bit-for-bit on the
reference stack (numpy 2.3.5, scikit-fmm 2025.6.23); manifest.sha256 pins the
digests so silent drift is caught in CI (same doctrine as
benchmarks/fixtures on the COBOL nonlinear-filter track).

Usage: python3 generate_fmm_goldens.py            (from this directory)
"""
import hashlib
import json
import os

import numpy as np
import skfmm

HERE = os.path.dirname(os.path.abspath(__file__))
NPY_VERSION = (1, 0)  # informational; small arrays always write .npy v1.0


def smooth_base(n, bumps):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    field = np.zeros((n, n))
    for (cx, cy, sigma, amp) in bumps:
        field += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / sigma)
    return field


def case_point_source_constant():
    n = 64
    phi = np.ones((n, n))
    phi[n // 2, n // 2] = -1.0
    return phi, np.ones((n, n)), 0.01, 2


def case_smooth_hills():
    n = 128
    speed = 0.5 + smooth_base(n, [(40.0, 70.0, 900.0, 0.8),
                                  (90.0, 35.0, 1600.0, 1.2)]) + 0.002 * np.mgrid[0:n, 0:n][1]
    phi = np.ones((n, n))
    phi[n // 2, n // 2] = -1.0
    return phi, speed, 0.01, 2


def case_rough_basin_masked():
    n = 128
    rng = np.random.default_rng(20261031)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    # Two-lobe basin analog of the pipeline's per-basin mask.
    basin = (((xx - 64.0) / 38.0) ** 2 + ((yy - 70.0) / 30.0) ** 2 < 1.0) | \
            (((xx - 30.0) / 20.0) ** 2 + ((yy - 40.0) / 18.0) ** 2 < 1.0)
    # Rough reciprocal-local-cost analog: bumpy, clamped to >= 1.0 like
    # Local_Cost_Computation's reciprocalLocalCostMinimum floor.
    speed_in = 1.0 + smooth_base(n, [(70.0, 60.0, 1200.0, 0.6),
                                     (35.0, 45.0, 500.0, 0.4)]) \
        + 0.05 * rng.random((n, n))
    speed = np.zeros((n, n))
    speed[basin] = np.maximum(speed_in[basin], 1.0)
    # Interior wall of zero-speed cells: exercises mask-skipping in the
    # neighbor scan (cells behind it route around, none stay unreachable).
    speed[58:62, 30:44] = 0.0
    phi = np.zeros((n, n))
    phi[basin] = 1.0
    start = (70, 64)
    assert basin[start], "start point must lie inside the basin"
    phi[start] = -1.0
    return phi, speed, 0.01, 2


def case_edge_start_rough():
    n = 96
    rng = np.random.default_rng(20261101)
    speed = 1.0 + 0.3 * rng.random((n, n)) ** 2
    phi = np.ones((n, n))
    phi[10, 10] = -1.0
    return phi, speed, 0.01, 2


def case_anisotropic_dx():
    n = 64
    speed = 1.0 + smooth_base(n, [(32.0, 32.0, 800.0, 0.5)])
    phi = np.ones((n, n))
    phi[n // 2, n // 2] = -1.0
    return phi, speed, (2.0, 1.0), 2


def case_row_1d():
    phi = np.array([[-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    speed = np.array([[2.0, 2.0, 1.0, 1.0, 0.5, 0.5, 2.0, 2.0]])
    return phi, speed, 1.0, 2


def case_order1_smooth():
    n = 64
    speed = 1.0 + 0.25 * smooth_base(n, [(40.0, 30.0, 700.0, 1.0)])
    phi = np.ones((n, n))
    phi[n // 2, n // 2] = -1.0
    return phi, speed, 0.01, 1


CASES = {
    "point_source_constant": case_point_source_constant,
    "smooth_hills": case_smooth_hills,
    "rough_basin_masked": case_rough_basin_masked,
    "edge_start_rough": case_edge_start_rough,
    "anisotropic_dx": case_anisotropic_dx,
    "row_1d": case_row_1d,
    "order1_smooth": case_order1_smooth,
}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    manifest = []
    for name, build in CASES.items():
        case_dir = os.path.join(HERE, name)
        os.makedirs(case_dir, exist_ok=True)
        phi, speed, dx, order = build()
        t = skfmm.travel_time(phi, speed, dx=dx, order=order)
        mask = np.zeros(phi.shape, dtype=bool)
        if isinstance(t, np.ma.MaskedArray):
            mask = np.array(t.mask, dtype=bool) if t.mask is not np.ma.nomask else mask
            t_data = np.asarray(t.data)
        else:
            t_data = np.asarray(t)
        np.save(os.path.join(case_dir, "phi.npy"), np.asarray(phi, dtype=np.float64))
        np.save(os.path.join(case_dir, "speed.npy"), np.asarray(speed, dtype=np.float64))
        np.save(os.path.join(case_dir, "travel_time.npy"), t_data)
        np.save(os.path.join(case_dir, "mask.npy"), mask)
        dx_str = ",".join(str(d) for d in dx) if isinstance(dx, tuple) else str(dx)
        with open(os.path.join(case_dir, "params.txt"), "w") as f:
            f.write(f"dx={dx_str}\norder={order}\n")
        meta = {
            "dx": list(dx) if isinstance(dx, tuple) else dx,
            "order": order,
            "skfmm_version": skfmm.__version__,
            "numpy_version": np.__version__,
            "masked_cells": int(mask.sum()),
        }
        with open(os.path.join(case_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
            f.write("\n")
        for fname in ("phi.npy", "speed.npy", "travel_time.npy", "mask.npy"):
            manifest.append(os.path.join(name, fname))
        print(f"{name}: wrote 4 arrays (masked cells: {int(mask.sum())})")
    manifest.sort()
    with open(os.path.join(HERE, "manifest.sha256"), "w") as f:
        for rel in manifest:
            f.write(f"{sha256(os.path.join(HERE, rel))}  {rel}\n")
    print(f"manifest.sha256: {len(manifest)} files pinned")


if __name__ == "__main__":
    main()
