# GeoNet algorithm specification

Technical spec of the channel-extraction algorithm as implemented on `master` (October 2026). Grounded in the six stage scripts and `pygeonet_defaults.py`; the parameter catalog at the end is exhaustive over that file. Related reading: [ARCHITECTURE.md](../ARCHITECTURE.md) (system shape), [docs/usage.md](usage.md) (CLIs and outputs).

The method follows Passalacqua et al. (2010) and Sangireddy et al. (2016): smooth the DEM with a nonlinear (edge-preserving) filter, route flow over it with GRASS GIS `r.watershed`, skeletonize the channelized area from flow-accumulation and curvature thresholds, then find channel heads as upstream endpoints of a geodesic travel-time field computed with the fast marching method.

## Stage 1 — Perona-Malik nonlinear diffusion

**Script:** `pygeonet_nonlinear_filter.py` · **Reads:** input DEM (`GeoInputs/GIS/<project>/<dem>.tif`) · **Writes:** `PM_filtered_grassgis.tif`

1. Read the DEM with GDAL (`pygeonet_rasterio.read_dem_from_geotiff`); pixels below `defaults.demNanFlag` (−9998.0) become NaN.
2. Compute the edge threshold λ (lambda): the `demSmoothingQuantile` (default 0.9) quantile of the absolute gradient magnitude of the raw DEM (`np.gradient` with the DEM's pixel scale), over non-NaN pixels, via `scipy.stats.mstats.mquantiles`. λ becomes the kappa (conductance) parameter of the diffusion.
3. Iterate `nFilterIterations` (default 50) explicit Perona-Malik updates over the NaN-aware image:

```text
gS = 1 / (1 + (δS/κ)²) / step      (option 2 — "PeronaMalik2", the default)
gS = exp(−(δS/κ)²) / step          (option 1 — "PeronaMalik1")
NS, EW = flux differences (south/north, east/west)
imgout += γ · (NS + EW)            γ = diffusionTimeIncrement = 0.1
```

- `diffusionMethod` selects the conductance form: `PeronaMalik2` (rational, default), `PeronaMalik1` (exponential). `Tukey` and `rampPreserving` appear in the defaults comments but are **not implemented** — selecting them prints "not available" and skips filtering.
- NaN cells: flux into a NaN cell is zeroed after each update (the NaN mask is reapplied), so NoData does not bleed.
- `diffusionTimeIncrement = 0.1` keeps the explicit scheme stable (the defaults file's own comment).
- `diffusionSigmaSquared = 0.05` parameterizes a Gaussian-smoothing helper (`simple_gaussian_smoothing`) that the main path does not call — smoothing happens through the diffusion itself.
- The filtered DEM is written as Float32 with the DEM's geotransform and a projection re-imported from its EPSG authority code.

## Stage 2 — Slope and curvature

**Script:** `pygeonet_slope_curvature.py` · **Reads:** `PM_filtered_grassgis.tif` · **Writes:** `<dem>_slope.tif`, `<dem>_curvature.tif`

- **Slope:** magnitude of `np.gradient(filteredDEM, pixelScale)`; written as Float32; min/max/mean/stddev slope statistics printed.
- **Curvature** (the Laplacian of the DEM, computed one of two ways per `curvatureCalcMethod`):
  - `geometric` (default): normalize the gradient to unit length, then take the divergence — `∇·(∇h/|∇h|)`.
  - `laplacian`: divergence of the raw gradient.
  - NaNs in curvature are zeroed before stats; the curvature array is then NaN-masked back to the filtered-DEM mask.
- Division-by-zero warnings are suppressed (`np.seterr(divide='ignore', invalid='ignore')`).
- A quantile-quantile plot helper (`compute_quantile_quantile_curve`, statsmodels) exists but is not called by `main()`.

## Stage 3 — Flow routing via GRASS GIS

**Script:** `pygeonet_grass_py3.py` (session mechanics in [ARCHITECTURE.md](../ARCHITECTURE.md#grass-session-dispatch-stage-3)) · **Reads:** `PM_filtered_grassgis.tif` · **Writes:** `<dem>_fdr.tif` (Int32), `<dem>_fac.tif` (Float64), `<dem>_outlets.tif` (Float32), `<dem>_basins.tif` (Int16)

1. Import the filtered DEM into the GRASS location (`r.in.gdal`, layer named after the DEM).
2. `r.watershed` — D8 flow routing with `threshold=thresholdAreaSubBasinIndexing` (1500 cells) for subbasin indexing:
   - DEMs larger than 4000 × 4000 pixels run `r.watershed -am` twice (drainage, then accumulation, using swap memory); smaller DEMs run `-a` once with both outputs.
   - Outputs: `dra1v23` (flow direction) and `acc1v23` (flow accumulation). This is **D8** flow direction (GRASS `r.watershed` convention: least-cost path on the 8-neighbor graph, flow split at the cell level resolved to one direction).
3. Outlets: `r.mapcalc outletmap = if(dra1v23 >= 0, null(), 1)` — every cell whose flow direction is negative (a GRASS-negative code for a pit/edge outflow) becomes an outlet; converted to points with `r.to.vect`.
4. Basins: `r.stream.basins` delineates subbasins upstream of the outlet points (addon installed at runtime via `g.extension`).
5. All four rasters are exported as GeoTIFFs with the listed dtypes.

The `thresholdAreaSubBasinIndexing` value is in **cells**: it is the smallest drainage area (in grid cells) GRASS will index as a separate subbasin.

## Stage 4 — Skeletonization by dual threshold

**Script:** `pygeonet_skeleton_definition.py` · **Reads:** `<dem>_curvature.tif`, `<dem>_fac.tif`, filtered DEM (mask) · **Writes:** `<dem>_flowskeleton.tif`, `<dem>_curvatureskeleton.tif`, `<dem>_skeleton.tif`

Three binary skeletons (1 = channel cell, 0 elsewhere):

- **Flow skeleton:** `compute_skeleton_by_single_threshold(flow, flowThresholdForSkeleton)` — cells with D8 accumulation area > 3000 cells (default). *Fixed by PR #10:* the `np.warnings.filterwarnings('ignore')` call (an `AttributeError` on NumPy ≥ 2.0) was replaced with the stdlib `warnings` module; the function runs on modern NumPy.
- **Curvature skeleton:** single threshold at `mean + 1.5·stddev` of the curvature distribution — the 1.5 sigma factor (`thresholdCurvatureQQxx`) is hardcoded in `main()` (the `defaults.thresholdQqCurvature = 0` knob is not wired).
- **Dual skeleton:** `mask_flow · mask_curvature` — cells above **both** thresholds. This is the skeleton the rest of the pipeline uses (`<dem>_skeleton.tif`).

## Stage 5 — Cost function and fast-marching travel time

**Script:** `pygeonet_fast_marching.py` · **Reads:** `<dem>_outlets.tif`, `<dem>_basins.tif`, `<dem>_curvature.tif`, `<dem>_fac.tif`, `<dem>_skeleton.tif`, filtered DEM · **Writes:** `<dem>_costfunction.tif`, `<dem>_geodesicDistance.tif`

### Basin qualification (numba)

A subbasin qualifies for channel-head search when both hold:

- its pixel count exceeds `thresholdPercentAreaForDelineation` (0.1) **percent** of the whole DEM, and
- it has more than `numBasinsElements` (2) pixels.

`Fast_Marching_Start_Point_Identification` evaluates this per label in a `@njit(parallel=True)` pass writing a disjoint keep-mask; `fmm_list_creation` reduces it deterministically in label order (PR #2 made this race-free and deterministic; the original appended to shared Python lists inside `prange`).

### Reciprocal local cost function

The cost field is the **speed** of the eikonal solver (higher = cheaper to traverse). `reciprocalLocalCostFn` in `pygeonet_defaults.py` is a **Python expression string evaluated with `eval()`** at stage 5, with exactly four names in scope:

| Name | Meaning |
|---|---|
| `flowArray` | D8 flow-accumulation area (`<dem>_fac.tif`) |
| `flowMean` | mean of `flowArray` over non-NaN cells |
| `skeletonFromFlowAndCurvatureArray` | the dual skeleton from stage 4 |
| `curvatureDemArray` | stage-2 curvature, normalized to [0, 1] (`doNormalizeCurvature=1`), NaNs zeroed |

The commented alternatives in `pygeonet_defaults.py` — `(flowArray.^(1/3)).*(slopeDemArray.^(-1/3)) + 20`, `flowArray + 30*skeletonFromFlowAndCurvatureArray + exp(curvatureDemArray*3)` — are historical; the names they reference (`areaArray`, `slopeDemArray`, `skeletonFromFlowArray`, `skeletonFromCurvatureArray`, `areaMean`) are **not in scope** at eval time. The active default is:

```text
flowArray + flowMean*skeletonFromFlowAndCurvatureArray + flowMean*curvatureDemArray
```

`reciprocalLocalCostMinimum = 'nan'` disables the optional floor that would clamp cells below a minimum cost to 1.0. The evaluated cost is written as `<dem>_costfunction.tif`.

### Travel-time solve (per basin)

For each qualifying start point (row, col from the outlet raster):

```text
phi   = 1 where flow accumulation ≠ 0 (the channel network), else 0
phi[start] = −1                                  # the source point
speed = reciprocal local cost field
travelTime = skfmm.travel_time(phi, speed, dx=0.01)
```

- `dx=0.01` is fixed: travel-time units are cost-weighted, not meters.
- The solve is wrapped by `travel_time_with_domain_guard`, which re-raises scikit-fmm `ValueError`/`RuntimeError` as `FastMarchingDomainError` (catchment index + start point attached). Background: scikit-fmm < 2021.1.20 aborts the march with "Negative discriminant in time marcher quadratic" on rough cost fields (scikit-fmm issue #18); the env pins 2021.7.8, which handles the common case internally.
- Each solved basin's travel times are scattered into the full-raster `geodesicDistanceArray` inside the loop (restored by PR #4 — after a refactor, only the final basin's distances survived).
- `∞` (unreached cells) → NaN; result written as `<dem>_geodesicDistance.tif`.
- `psutil` prints RAM usage per solve; `dask.distributed` is imported but unused on the current serial path. Order-of-magnitude runtime on a 0.7 GB DEM: 5–7 minutes.

## Stage 6 — Channel-head detection

**Script:** `pygeonet_channel_head_definition.py` · **Reads:** `<dem>_skeleton.tif`, `<dem>_geodesicDistance.tif` · **Writes:** `<dem>_channelHeads.tif`, `<dem>_channelHeads.shp`

> *Fixed by PR #10:* the module-level `from time import clock` (an `ImportError` on Python ≥ 3.8) was removed; the stage imports on any Python 3. The pinned conda env (Python 3.7.7) remains the verified end-to-end baseline.

Algorithm — a channel head is a skeleton endpoint that is a local **maximum** of geodesic travel time along its connected component:

1. Label 8-connected skeleton components (`scipy.ndimage.label` with a 3×3 structuring element).
2. Count each component's pixels; build a histogram of those counts (`bins='auto'`).
3. Exclude small components: the threshold is derived from the histogram counts above the 0.3 quantile (hardcoded). Components smaller than that count are not searched. This is the "0.3" the README's channel-head note refers to.
4. Scan every remaining skeleton cell; in a search box of `endPointSearchBoxSize` = 30 pixels around it (clipped to DEM bounds), look for another cell of the **same component** with a strictly greater geodesic distance. If none exists, the cell is a channel head.
5. Heads are written as a point shapefile (`pygeonet_vectorio.write_drainage_nodes`, EPSG from the DEM) and a 0/1 raster.

The box size 30 is documented in defaults as "median hillslope of the input DEM — the distance between each pixel and the first channelized downslope pixel".

## Parameter catalog — `pygeonet_defaults.py`

Every parameter in the file, its default, and its effect. "wired" = actually consumed by the pipeline on `master`.

### Reporting / plotting toggles

| Parameter | Default | Wired | Effect |
|---|---|---|---|
| `doFileOutput` | 1 | No | Intent: control file output. No stage reads it. |
| `doReport` | 1 | No | Intent: control reporting. No stage reads it. |
| `doPlot` | 1 | Yes | Guards `pygeonet_plot.raster_*_plot` calls (mostly commented out in stage `main()`s). |
| `doResetFiguresOnStart` | 1 | No | Intent: reset matplotlib figures. No stage reads it. |

### Stage 1 — diffusion

| Parameter | Default | Wired | Effect |
|---|---|---|---|
| `diffusionMethod` | `'PeronaMalik2'` | Yes | Conductance form: `PeronaMalik1` (exponential) or `PeronaMalik2` (rational). `Tukey` / `rampPreserving` are not implemented. |
| `diffusionTimeIncrement` | 0.1 | Yes | Explicit-scheme step γ; must stay small for stability. |
| `diffusionSigmaSquared` | 0.05 | Yes* | Variance for `simple_gaussian_smoothing` — defined, but that helper is not called by `main()`. |
| `nFilterIterations` | 50 | Yes | Number of Perona-Malik iterations. |
| `demSmoothingQuantile` | 0.9 | Yes | Quantile of |∇DEM| used as edge threshold λ. Good range 0.5–0.9 (defaults comment): smooth the lower (1−q) of the landscape, enhance the top q. |

### Stage 2 — curvature

| Parameter | Default | Wired | Effect |
|---|---|---|---|
| `curvatureCalcMethod` | `'geometric'` | Yes | `geometric`: divergence of unit gradient; `laplacian`: divergence of raw gradient. |
| `thresholdQqCurvature` | 0 | No | Name suggests a QQ-plot curvature threshold; no stage reads it. |

### Stage 3 — routing / basins

| Parameter | Default | Wired | Effect |
|---|---|---|---|
| `thresholdAreaSubBasinIndexing` | 1500 | Yes | `r.watershed threshold=`: minimum subbasin size in cells for basin indexing. Also gates the >4000-px swap-memory path. |
| `demNanFlag` | −9998.0 | Yes | DEM values below this become NaN (NoData). |
| `demErrorFlag` | −9998.0 | No | Twin of `demNanFlag` kept for GeoFlood parity; no stage reads it. |

### Stage 4 — skeletons

| Parameter | Default | Wired | Effect |
|---|---|---|---|
| `flowThresholdForSkeleton` | 3000 | Yes | Flow-accumulation (cells) above which a cell is channel in the flow skeleton and the dual skeleton. |

(The curvature-skeleton threshold `mean + 1.5σ` is hardcoded in `pygeonet_skeleton_definition.py`, not in defaults.)

### Stage 5 — cost / fast marching

| Parameter | Default | Wired | Effect |
|---|---|---|---|
| `reciprocalLocalCostFn` | `'flowArray + flowMean*skeletonFromFlowAndCurvatureArray + flowMean*curvatureDemArray'` | Yes | Cost expression, `eval()`-ed with `flowArray`, `flowMean`, `skeletonFromFlowAndCurvatureArray`, `curvatureDemArray` in scope. |
| `doNormalizeCurvature` | 1 | Yes | Min-max normalize curvature to [0, 1] before cost evaluation. |
| `reciprocalLocalCostMinimum` | `'nan'` | Yes* | If not `'nan'`, cells with cost below it are clamped to 1.0. Default disables the floor. |
| `thresholdPercentAreaForDelineation` | 0.1 | Yes | Minimum basin size as **percent of DEM pixels** (0.1%) to qualify for channel-head search. |
| `numBasinsElements` | 2 | Yes | Minimum basin pixel count (defined in `pygeonet_prepare.py`). |

### Stage 6 — channel heads

| Parameter | Default | Wired | Effect |
|---|---|---|---|
| `endPointSearchBoxSize` | 30 | Yes | Search-box half-size (pixels) for the local-maximum test; documented as median hillslope. |
| `channelheadPredefined` | 0 | No | Name suggests support for user-supplied heads; no stage reads it. |
| `doTrueGradientDescent` | 1 | No | Comment says it selects discrete geodesic path-finding; no stage reads it. |

### Defined elsewhere but unwired

| Name | Defined in | Value | Status |
|---|---|---|---|
| `split_distance` | `pygeonet_prepare.py` | 1000 | No stage reads it. |
| `thresholdCurvatureQQxx` | `pygeonet_skeleton_definition.py` (`main()` local) | 1.5 | The σ-multiplier for the curvature-skeleton threshold — hardcoded, not a defaults knob. |

> Tuning guidance: the two thresholds that most change results are `flowThresholdForSkeleton` (3000 cells — "how much water makes a channel") and `endPointSearchBoxSize` (30 px — how far uphill a head candidate must dominate). `demSmoothingQuantile` (0.9) controls how aggressively stage 1 smooths valleys.
