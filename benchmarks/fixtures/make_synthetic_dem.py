"""Regenerate the committed synthetic DEM fixture (deterministic).

The fixture is a 128x128 float32 DEM-like raster (two ridges, a planar
gradient, and seeded noise) used by the COBOL parity tests and the
nonlinear-filter benchmark. Regenerating must reproduce the committed
.npy bit-for-bit on any platform; the parity test asserts the committed
sha256 so silent drift is caught in CI.
"""
import hashlib
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'synthetic_dem_128x128.npy')


def build():
    rng = np.random.default_rng(20261030)
    n = 128
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    dem = (30.0
           + 12.0 * np.exp(-((xx - 40.0) ** 2 + (yy - 72.0) ** 2) / 900.0)
           + 22.0 * np.exp(-((xx - 90.0) ** 2 + (yy - 30.0) ** 2) / 1600.0)
           + 0.08 * xx
           + 0.05 * yy
           + rng.normal(0.0, 0.15, size=(n, n)))
    return dem.astype(np.float32)


def main():
    dem = build()
    np.save(OUT, dem)
    with open(OUT, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"wrote {OUT}")
    print(f"sha256: {digest}")


if __name__ == '__main__':
    main()
