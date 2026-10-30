// SPDX-License-Identifier: GPL-3.0-or-later
//! Stage-6 channel-head scan: a byte-exact port of the compute core of
//! `pygeonet_channel_head_definition.Channel_Head_Definition`.
//!
//! Serial full-raster traversal (no parallelism: any reordering risks the A1
//! byte-match gate), fixed 8-connectivity connected-component labeling, and
//! NumPy 1.19.1-exact histogram/quantile arithmetic -- the oracle environment
//! of the archived e7e1bdb golden set. Never writes files; plotting and
//! raster/vector writes stay Python-side (pygeonet_channel_head_definition).
//!
//! Faithfully preserved quirks (each is observable in the Python path):
//! - the search-box slice spans ``[i-yMinus, i+yPlus)`` with
//!   ``yMinus = min(w, i-1)`` and ``yPlus = min(w, nrows-i)`` (``w`` is the
//!   ``pygeonet_defaults.endPointSearchBoxSize`` constant), so row 0 / col 0
//!   pixels are never neighbours inside a window;
//! - the same-label mask references ``w[i-r0, j-c0]`` = the window element at
//!   ``(yMinus, xMinus)`` with PYTHON negative indexing: for row 0 / col 0
//!   pixels (``yMinus == -1``) the reference is the last window row, not the
//!   center pixel, so the "same component" test compares against a different
//!   pixel's label (possibly background);
//! - the distance test ``skeleton * box > d_ij`` multiplies in the promoted
//!   input dtype (f32 x f32 stays f32) and NaN distances never compare true;
//! - the 0.3-quantile threshold test runs on the histogram BIN EDGES (the
//!   upstream variable ``skeletonNumElementsHistogramX`` is the edges array;
//!   the histogram counts are computed but never used);
//! - post-#14 degenerate semantics: the threshold falls back to the quantile
//!   value when no edge qualifies; a component-free (empty) mask skips the
//!   size statistics and a scan that finds no heads both return defined
//!   empty coordinate outputs;
//! - a 1-row/1-column raster with a qualifying pixel still raises IndexError
//!   at the empty-window reference lookup (the surviving Python quirk).

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyModule};

type HeadCoordinates = (Py<PyArray1<isize>>, Py<PyArray1<isize>>);

/// Arithmetic in the promoted input dtype: ``v * box > d`` with a real
/// multiply so NaN propagates exactly as numpy's elementwise expression does.
trait NativeFloat: Copy {
    fn product_gt(skeleton: Self, distance: Self, d_ij: Self) -> bool;
}

impl NativeFloat for f32 {
    fn product_gt(skeleton: Self, distance: Self, d_ij: Self) -> bool {
        skeleton * distance > d_ij
    }
}

impl NativeFloat for f64 {
    fn product_gt(skeleton: Self, distance: Self, d_ij: Self) -> bool {
        skeleton * distance > d_ij
    }
}

/// Both input rasters flattened row-major in the promoted comparison dtype.
/// f32 x f32 stays f32; anything involving f64 or a wide integer widens to
/// f64 (legacy numpy value-based promotion: i32 + f32 -> f64, u8 + f32 -> f32).
enum PromotedGrids {
    F32 {
        skeleton: Vec<f32>,
        distances: Vec<f32>,
    },
    F64 {
        skeleton: Vec<f64>,
        distances: Vec<f64>,
    },
}

/// Skeleton raster values cast to f32 (narrow sources: f32/bool/u8).
fn skeleton_values_f32(skeleton: &Bound<'_, PyUntypedArray>) -> Option<Vec<f32>> {
    macro_rules! try_dtype {
        ($ty:ty, $conv:expr) => {
            if let Ok(array) = skeleton.cast::<PyArray2<$ty>>() {
                let view = array.readonly();
                let values = view
                    .as_array()
                    .iter()
                    .map(|&v| $conv(v))
                    .collect::<Vec<f32>>();
                return Some(values);
            }
        };
    }
    try_dtype!(f32, |v: f32| v);
    try_dtype!(bool, |v: bool| if v { 1.0f32 } else { 0.0f32 });
    try_dtype!(u8, |v: u8| v as f32);
    None
}

/// Skeleton raster values cast to f64 (wide sources; every value is exactly
/// representable at f64 width).
fn skeleton_values_f64(skeleton: &Bound<'_, PyUntypedArray>) -> Option<Vec<f64>> {
    macro_rules! try_dtype {
        ($ty:ty, $conv:expr) => {
            if let Ok(array) = skeleton.cast::<PyArray2<$ty>>() {
                let view = array.readonly();
                let values = view
                    .as_array()
                    .iter()
                    .map(|&v| $conv(v))
                    .collect::<Vec<f64>>();
                return Some(values);
            }
        };
    }
    try_dtype!(f64, |v: f64| v);
    try_dtype!(f32, |v: f32| v as f64);
    try_dtype!(bool, |v: bool| if v { 1.0f64 } else { 0.0f64 });
    try_dtype!(u8, |v: u8| v as f64);
    try_dtype!(i32, |v: i32| v as f64);
    try_dtype!(i64, |v: i64| v as f64);
    None
}

/// select_skeleton_size_threshold (post-#14): the first edge strictly above
/// the edges' 0.3-quantile; when the strict selection is empty -- all-equal
/// edges -- the quantile value itself is the threshold.
fn select_size_threshold(edges: &[f64]) -> f64 {
    let mut sorted = edges.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("edges are finite"));
    let q = quantile_linear(&sorted, 0.3);
    edges.iter().copied().find(|&edge| edge > q).unwrap_or(q)
}

fn distance_values_f32(distances: &Bound<'_, PyUntypedArray>) -> Option<Vec<f32>> {
    if let Ok(array) = distances.cast::<PyArray2<f32>>() {
        let view = array.readonly();
        return Some(view.as_array().iter().copied().collect::<Vec<f32>>());
    }
    None
}

fn distance_values_f64(distances: &Bound<'_, PyUntypedArray>) -> Option<Vec<f64>> {
    if let Ok(array) = distances.cast::<PyArray2<f64>>() {
        let view = array.readonly();
        return Some(view.as_array().iter().copied().collect::<Vec<f64>>());
    }
    if let Ok(array) = distances.cast::<PyArray2<f32>>() {
        let view = array.readonly();
        return Some(
            view.as_array()
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<f64>>(),
        );
    }
    None
}

impl PromotedGrids {
    /// Extract both rasters with matching 2-D shapes into the promoted dtype.
    fn extract<'py>(
        skeleton: &Bound<'py, PyUntypedArray>,
        distances: &Bound<'py, PyUntypedArray>,
    ) -> PyResult<(Self, usize, usize)> {
        let (sk_rows, sk_cols) = grid_shape(skeleton, "skeletonFromFlowAndCurvatureArray")?;
        let (d_rows, d_cols) = grid_shape(distances, "geodesicDistanceArray")?;
        if (sk_rows, sk_cols) != (d_rows, d_cols) {
            return Err(PyValueError::new_err(format!(
                "geodesicDistanceArray shape [{d_rows}, {d_cols}] does not match \
                 skeletonFromFlowAndCurvatureArray shape [{sk_rows}, {sk_cols}]"
            )));
        }
        // Promotion classes of the skeleton raster per the legacy numpy rules.
        macro_rules! skeleton_wide {
            ($ty:ty) => {
                skeleton.cast::<PyArray2<$ty>>().is_ok()
            };
        }
        let wide = skeleton_wide!(f64)
            || skeleton_wide!(i32)
            || skeleton_wide!(i64)
            || distances.cast::<PyArray2<f64>>().is_ok();
        if wide {
            let skel = skeleton_values_f64(skeleton)
                .ok_or_else(|| PyValueError::new_err("skeleton dtype not supported"))?;
            let dist = distance_values_f64(distances)
                .ok_or_else(|| PyValueError::new_err("geodesic dtype not supported"))?;
            Ok((
                PromotedGrids::F64 {
                    skeleton: skel,
                    distances: dist,
                },
                sk_rows,
                sk_cols,
            ))
        } else {
            let skel = skeleton_values_f32(skeleton)
                .ok_or_else(|| PyValueError::new_err("skeleton dtype not supported"))?;
            let dist = distance_values_f32(distances)
                .ok_or_else(|| PyValueError::new_err("geodesic dtype not supported"))?;
            Ok((
                PromotedGrids::F32 {
                    skeleton: skel,
                    distances: dist,
                },
                sk_rows,
                sk_cols,
            ))
        }
    }

    fn nrows(&self) -> usize {
        match self {
            PromotedGrids::F32 { skeleton, .. } => skeleton.len(),
            PromotedGrids::F64 { skeleton, .. } => skeleton.len(),
        }
    }
}

fn grid_shape(array: &Bound<'_, PyUntypedArray>, name: &str) -> PyResult<(usize, usize)> {
    let shape = array.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "{name} must be 2-D, got shape {shape:?}"
        )));
    }
    Ok((shape[0], shape[1]))
}

/// 8-connectivity component labeling (scipy.ndimage.label with a 3x3 ones
/// structure). Label values are assigned in first-encounter scan order; the
/// scan result is invariant to any consistent relabeling because it only
/// compares labels for equality and uses per-label counts.
fn label_components(foreground: &[bool], nrows: usize, ncols: usize) -> Vec<i32> {
    let mut labels = vec![0i32; nrows * ncols];
    let mut next_label = 0i32;
    for start in 0..nrows * ncols {
        if !foreground[start] || labels[start] != 0 {
            continue;
        }
        next_label += 1;
        let mut stack = vec![start];
        labels[start] = next_label;
        while let Some(pixel) = stack.pop() {
            let row = pixel / ncols;
            let col = pixel % ncols;
            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let r = row as i64 + dr;
                    let c = col as i64 + dc;
                    if r < 0 || c < 0 || r >= nrows as i64 || c >= ncols as i64 {
                        continue;
                    }
                    let neighbor = r as usize * ncols + c as usize;
                    if foreground[neighbor] && labels[neighbor] == 0 {
                        labels[neighbor] = next_label;
                        stack.push(neighbor);
                    }
                }
            }
        }
    }
    labels
}

/// numpy 1.19.1 ``np.linspace(start, stop, num, endpoint=True)`` for num >= 2:
/// y = arange(num) * delta; y /= (num - 1); y += start; y[-1] = stop.
/// The operation order is load-bearing (delta and division before the offset).
fn linspace_edges(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let delta = stop - start;
    let div = (num - 1) as f64;
    let mut y: Vec<f64> = (0..num).map(|k| (k as f64) * delta / div + start).collect();
    if num > 1 {
        y[num - 1] = stop;
    }
    y
}

/// numpy 1.19.1 ``np.percentile(a, q)`` with linear interpolation:
/// indices = q * (n - 1); below = floor; above = min(below + 1, n - 1);
/// gamma = indices - below; result = a[below] * (1 - gamma) + a[above] * gamma.
/// ``a`` must be sorted and finite (stage-6 inputs never carry NaN here).
fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let virtual_index = q * ((n - 1) as f64);
    let below = virtual_index.floor() as usize;
    let above = (below + 1).min(n - 1);
    let gamma = virtual_index - below as f64;
    sorted[below] * (1.0 - gamma) + sorted[above] * gamma
}

/// numpy 1.19.1 ``np.median``: middle element, or the mean of the two middle
/// elements for even lengths.
fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    match n % 2 {
        1 => sorted[n / 2],
        _ => (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
    }
}

/// numpy 1.19.1 ``np.histogram(counts, bins='auto')`` bin edges. Stage 6
/// thresholds on the edges only (the upstream histogram counts variable is
/// dead), so the counting phase is not ported.
fn histogram_auto_edges(counts: &[i64]) -> Vec<f64> {
    if counts.is_empty() {
        // _get_outer_edges: size-0 data autodetects the range (0, 1);
        // _get_bin_edges: size-0 data uses one bin.
        return linspace_edges(0.0, 1.0, 2);
    }
    let min = *counts.iter().min().unwrap();
    let max = *counts.iter().max().unwrap();
    // _get_outer_edges: expand a degenerate range by half a unit.
    let (first_edge, last_edge, span) = if min == max {
        let first = min as f64 - 0.5;
        let last = max as f64 + 0.5;
        (first, last, 1.0f64)
    } else {
        let first = min as f64;
        let last = max as f64;
        (first, last, (max - min) as f64)
    };
    let n = counts.len();
    // _hist_bin_fd: iqr = percentile(x, 75) - percentile(x, 25)
    let mut sorted: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("counts are finite"));
    let iqr = quantile_linear(&sorted, 0.75) - quantile_linear(&sorted, 0.25);
    let fd_bw = 2.0 * iqr * (n as f64).powf(-1.0 / 3.0);
    // _hist_bin_sturges: ptp / (log2(n) + 1). ptp is the DATA range: for
    // constant data it is zero, so the width is zero and the histogram
    // collapses to a single bin over the expanded outer edges.
    let data_ptp = (max - min) as f64;
    let sturges_bw = data_ptp / ((n as f64).log2() + 1.0);
    // _hist_bin_auto: min width unless FD is zero; zero width -> one bin.
    let width = if fd_bw != 0.0 {
        fd_bw.min(sturges_bw)
    } else {
        sturges_bw
    };
    let n_equal_bins = if width > 0.0 {
        (span / width).ceil() as usize
    } else {
        1
    };
    linspace_edges(first_edge, last_edge, n_equal_bins + 1)
}

/// Mirror of Python's ``print`` (builtins.print, one argument).
fn py_print(py: Python<'_>, message: &str) -> PyResult<()> {
    py.import("builtins")?.getattr("print")?.call1((message,))?;
    Ok(())
}

/// str() of a Python float, used for the Max/Median/threshold lines so the
/// native path's console output matches the Python oracle's exactly.
fn py_float_str(py: Python<'_>, value: f64) -> PyResult<String> {
    let text = PyFloat::new(py, value).str()?.extract::<String>()?;
    Ok(text)
}

/// One window's distance test, ported line-for-line from the scan:
/// head-suppressed iff any window pixel satisfies
/// ``(skeleton * distance > d_ij) and (label == reference label)``.
/// The reference label is read with Python's negative-index semantics at
/// ``(yMinus, xMinus)`` inside the window.
struct WindowScan<'a, T: NativeFloat> {
    skeleton: &'a [T],
    distances: &'a [T],
    labels: &'a [i32],
    counts: &'a [i64],
    threshold: f64,
    nrows: usize,
    ncols: usize,
    search_box: i64,
}

impl<T: NativeFloat> WindowScan<'_, T> {
    fn run(
        &self,
        on_iteration: &mut dyn FnMut(usize) -> PyResult<()>,
    ) -> PyResult<Vec<(i64, i64)>> {
        let mut heads: Vec<(i64, i64)> = Vec::new();
        let nrows = self.nrows;
        let ncols = self.ncols;
        for i in 0..nrows {
            if i % 500 == 0 {
                on_iteration(nrows - i)?;
            }
            for j in 0..ncols {
                let label = self.labels[i * ncols + j];
                if label == 0 {
                    continue;
                }
                let pixel_count = self.counts[(label - 1) as usize] as f64;
                if pixel_count < self.threshold {
                    continue;
                }
                let y_minus = self.search_box.min(i as i64 - 1);
                let y_plus = self.search_box.min(nrows as i64 - i as i64);
                let x_minus = self.search_box.min(j as i64 - 1);
                let x_plus = self.search_box.min(ncols as i64 - j as i64);
                let win_rows = (y_plus + y_minus) as usize;
                let win_cols = (x_plus + x_minus) as usize;
                if win_rows == 0 || win_cols == 0 {
                    // Python indexes w[i-r0, j-c0] on an empty window axis:
                    // IndexError before any comparison happens.
                    return Err(PyIndexError::new_err(
                        "index -1 is out of bounds for axis 0 with size 0",
                    ));
                }
                let ref_row = if y_minus < 0 {
                    (win_rows as i64 + y_minus) as usize
                } else {
                    y_minus as usize
                };
                let ref_col = if x_minus < 0 {
                    (win_cols as i64 + x_minus) as usize
                } else {
                    x_minus as usize
                };
                let ref_label = self.labels[((i as i64 - y_minus) as usize + ref_row) * ncols
                    + ((j as i64 - x_minus) as usize + ref_col)];
                let d_ij = self.distances[i * ncols + j];
                let row_start = (i as i64 - y_minus) as usize;
                let row_end = (i as i64 + y_plus) as usize;
                let col_start = (j as i64 - x_minus) as usize;
                let col_end = (j as i64 + x_plus) as usize;
                let mut is_head = true;
                'window: for r in row_start..row_end {
                    for c in col_start..col_end {
                        if self.labels[r * ncols + c] != ref_label {
                            continue;
                        }
                        if T::product_gt(
                            self.skeleton[r * ncols + c],
                            self.distances[r * ncols + c],
                            d_ij,
                        ) {
                            is_head = false;
                            break 'window;
                        }
                    }
                }
                if is_head {
                    heads.push((i as i64, j as i64));
                }
            }
        }
        Ok(heads)
    }
}

/// The full stage-6 compute core over prepared grids. Returns the head
/// coordinates as (row, col) pairs in scan order.
fn channel_head_scan(
    py: Python<'_>,
    grids: &PromotedGrids,
    nrows: usize,
    ncols: usize,
    search_box: i64,
) -> PyResult<Vec<(i64, i64)>> {
    py_print(py, "Locating skeleton end points")?;
    // Component labeling over skeleton != 0 (NaN != 0 is foreground in numpy
    // too, and the promoted f32/f64 casts keep NaN non-zero).
    let foreground: Vec<bool> = match grids {
        PromotedGrids::F32 { skeleton, .. } => skeleton.iter().map(|&v| v != 0.0).collect(),
        PromotedGrids::F64 { skeleton, .. } => skeleton.iter().map(|&v| v != 0.0).collect(),
    };
    let labels = label_components(&foreground, nrows, ncols);
    let n_components = labels.iter().fold(0i32, |m, &l| m.max(l)) as usize;
    let mut counts = vec![0i64; n_components];
    for &label in &labels {
        if label != 0 {
            counts[(label - 1) as usize] += 1;
        }
    }
    py_print(
        py,
        "Counting the number of elements of each connected component",
    )?;
    if n_components == 0 {
        // Post-#14 degenerate skeleton: no components, zero heads, defined
        // empty outputs -- the size statistics are skipped entirely.
        py_print(
            py,
            "Skeleton has no connected components; no channel heads to find",
        )?;
        return Ok(Vec::new());
    }
    let edges = histogram_auto_edges(&counts);
    py_print(
        py,
        &format!("Max: {}", py_float_str(py, edges[edges.len() - 1])?),
    )?;
    let mut sorted_edges = edges.clone();
    sorted_edges.sort_by(|a, b| a.partial_cmp(b).expect("edges are finite"));
    py_print(
        py,
        &format!("Median: {}", py_float_str(py, median(&sorted_edges))?),
    )?;
    let threshold = select_size_threshold(&edges);
    py_print(
        py,
        &format!(
            "skeletonNumElementsThreshold: {}",
            py_float_str(py, threshold)?
        ),
    )?;
    let heads = match grids {
        PromotedGrids::F32 {
            skeleton,
            distances,
        } => WindowScan {
            skeleton,
            distances,
            labels: &labels,
            counts: &counts,
            threshold,
            nrows,
            ncols,
            search_box,
        }
        .run(&mut |remaining| py_print(py, &format!("Iterations Remaining: {remaining}")))?,
        PromotedGrids::F64 {
            skeleton,
            distances,
        } => WindowScan {
            skeleton,
            distances,
            labels: &labels,
            counts: &counts,
            threshold,
            nrows,
            ncols,
            search_box,
        }
        .run(&mut |remaining| py_print(py, &format!("Iterations Remaining: {remaining}")))?,
    };
    Ok(heads)
}

#[pyfunction]
#[allow(non_snake_case)]
pub fn find_channel_heads(
    py: Python<'_>,
    skeletonFromFlowAndCurvatureArray: &Bound<'_, PyUntypedArray>,
    geodesicDistanceArray: &Bound<'_, PyUntypedArray>,
) -> PyResult<HeadCoordinates> {
    let (grids, nrows, ncols) =
        PromotedGrids::extract(skeletonFromFlowAndCurvatureArray, geodesicDistanceArray)?;
    debug_assert_eq!(grids.nrows(), nrows * ncols);
    // The search-box size is a pygeonet_defaults config constant, read at
    // call time exactly as the Python function reads the module global.
    let defaults: Bound<'_, PyModule> = py.import("pygeonet_defaults")?;
    let search_box: i64 = defaults.getattr("endPointSearchBoxSize")?.extract()?;
    let heads = channel_head_scan(py, &grids, nrows, ncols, search_box)?;
    // Return construction, mirroring post-#14 Channel_Head_Definition:
    // zero heads produce defined empty coordinate arrays (the split halves
    // of np.zeros((2, 0), dtype=int)), matching the Python empty path.
    let mut xx: Vec<isize> = Vec::with_capacity(heads.len());
    let mut yy: Vec<isize> = Vec::with_capacity(heads.len());
    for &(row, col) in &heads {
        yy.push(row as isize);
        xx.push(col as isize);
    }
    Ok((xx.into_pyarray(py).unbind(), yy.into_pyarray(py).unbind()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Anchors observed from the oracle environment (numpy 1.19.1 semantics,
    // verified against the frozen-input probe and direct numpy calls).

    #[test]
    fn threshold_picks_first_edge_above_quantile() {
        // bench_m_1024 oracle anchor: q0.3 of the edges is 27.8, so the
        // threshold is the third edge.
        assert_eq!(
            select_size_threshold(&[2.0, 19.2, 36.4, 53.6, 70.8, 88.0]),
            36.4
        );
    }

    #[test]
    fn threshold_falls_back_to_quantile_when_no_edge_qualifies() {
        // All-equal edges: the strict selection is empty, so the quantile
        // value itself is the threshold (post-#14 review probe P3).
        assert_eq!(select_size_threshold(&[7.5, 7.5, 7.5, 7.5]), 7.5);
    }

    #[test]
    fn linspace_matches_numpy_1024_edges() {
        // bench_m_1024 oracle: counts span [2, 88], 5 bins.
        assert_eq!(
            linspace_edges(2.0, 88.0, 6),
            vec![2.0, 19.2, 36.4, 53.6, 70.8, 88.0]
        );
        assert_eq!(linspace_edges(0.0, 1.0, 2), vec![0.0, 1.0]);
        // Synthetic [1, 5, 20]: three bins via the Sturges width.
        assert_eq!(
            linspace_edges(1.0, 20.0, 4),
            vec![1.0, 7.333333333333333, 13.666666666666666, 20.0]
        );
    }

    #[test]
    fn quantile_linear_matches_numpy() {
        let edges = linspace_edges(2.0, 88.0, 6);
        let mut sorted = edges.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(quantile_linear(&sorted, 0.3), 27.799999999999997);
        assert_eq!(quantile_linear(&sorted, 0.0), 2.0);
        assert_eq!(quantile_linear(&sorted, 1.0), 88.0);
    }

    #[test]
    fn histogram_auto_edges_matches_numpy() {
        assert_eq!(histogram_auto_edges(&[]), vec![0.0, 1.0]);
        assert_eq!(histogram_auto_edges(&[88]), vec![87.5, 88.5]);
        assert_eq!(histogram_auto_edges(&[7, 7]), vec![6.5, 7.5]);
        // Verified against numpy 1.19.1 semantics via direct numpy calls.
        assert_eq!(
            histogram_auto_edges(&[1, 5, 20]),
            vec![1.0, 7.333333333333333, 13.666666666666666, 20.0]
        );
        assert_eq!(
            histogram_auto_edges(&[2, 2, 2, 88, 88]),
            vec![2.0, 23.5, 45.0, 66.5, 88.0]
        );
    }

    #[test]
    fn labeling_merges_diagonal_neighbors() {
        // 8-connectivity: two diagonally touching pixels are one component.
        let foreground = vec![true, false, false, true];
        assert_eq!(label_components(&foreground, 2, 2), vec![1, 0, 0, 1]);
        // Adjacent pixels share a label.
        let foreground = vec![true, true, false, false];
        assert_eq!(label_components(&foreground, 2, 2), vec![1, 1, 0, 0]);
        // A diagonal chain IS one component under 8-connectivity: (0,0),
        // (1,0), (1,1), (2,2) all touch pairwise (the last diagonally).
        let foreground = vec![true, false, false, true, true, false, false, false, true];
        assert_eq!(
            label_components(&foreground, 3, 3),
            vec![1, 0, 0, 1, 1, 0, 0, 0, 1]
        );
        // Spatially separated pixels (no 8-neighborhood contact) get
        // distinct scan-order labels.
        let foreground = vec![true, false, false, false, false, true, true, false, false];
        assert_eq!(
            label_components(&foreground, 3, 3),
            vec![1, 0, 0, 0, 0, 2, 3, 0, 0]
        );
    }

    #[test]
    fn scan_finds_single_component_head() {
        // A 5-pixel vertical component at col 2 (col >= 1, so the reference
        // lookup lands on the center pixel itself), distances increasing
        // downward: only the bottom pixel has no greater same-label
        // neighbour in its window.
        let mut skeleton = vec![0.0f32; 40 * 4];
        let mut distances = vec![0.0f32; 40 * 4];
        for r in 2..7 {
            skeleton[r * 4 + 2] = 1.0;
            distances[r * 4 + 2] = r as f32;
        }
        let labels = label_components(
            &skeleton.iter().map(|&v| v != 0.0).collect::<Vec<bool>>(),
            40,
            4,
        );
        let counts = vec![5i64];
        let scan = WindowScan {
            skeleton: &skeleton,
            distances: &distances,
            labels: &labels,
            counts: &counts,
            threshold: 0.0,
            nrows: 40,
            ncols: 4,
            search_box: 30,
        };
        let heads = scan.run(&mut |_| Ok(())).unwrap();
        assert_eq!(heads, vec![(6, 2)]);
    }

    #[test]
    fn scan_zero_width_window_is_indexerror() {
        // A 1-row raster: y_plus + y_minus == 0, so the reference lookup on
        // the empty window axis raises IndexError like the Python path.
        let skeleton = vec![1.0f32; 1];
        let distances = vec![1.0f32; 1];
        let labels = vec![1i32; 1];
        let counts = vec![1i64];
        let scan = WindowScan {
            skeleton: &skeleton,
            distances: &distances,
            labels: &labels,
            counts: &counts,
            threshold: 0.0,
            nrows: 1,
            ncols: 1,
            search_box: 30,
        };
        let error = scan.run(&mut |_| Ok(())).unwrap_err();
        assert_eq!(
            error.to_string(),
            "IndexError: index -1 is out of bounds for axis 0 with size 0"
        );
    }

    #[test]
    fn scan_row_zero_reference_wrap_quirk() {
        // On row 0 the same-label mask references the LAST window row
        // (w[-1]), not the center pixel. Build a 2-row grid where the
        // qualifying row-0 pixel's reference label is background (0): then no
        // window pixel can share the reference label and the pixel is a head
        // even though a same-component neighbour would otherwise suppress it.
        let ncols = 4;
        let nrows = 40;
        let mut skeleton = vec![0.0f32; nrows * ncols];
        let mut distances = vec![0.0f32; nrows * ncols];
        // One component spanning row 0 and row 1 at column 1.
        for r in [0usize, 1usize] {
            skeleton[r * ncols + 1] = 1.0;
            distances[r * ncols + 1] = 5.0;
        }
        let labels = label_components(
            &skeleton.iter().map(|&v| v != 0.0).collect::<Vec<bool>>(),
            nrows,
            ncols,
        );
        let counts = vec![2i64];
        let scan = WindowScan {
            skeleton: &skeleton,
            distances: &distances,
            labels: &labels,
            counts: &counts,
            threshold: 0.0,
            nrows,
            ncols,
            search_box: 30,
        };
        let heads = scan.run(&mut |_| Ok(())).unwrap();
        // Window for (0, 1): rows 1..30, cols 1..4 -> reference = w[-1, 0] =
        // label at (29, 1) = 0 (background) -> w1 matches only background
        // pixels, whose skeleton*distance = 0*5 = 0 > 5 is false -> head.
        assert!(heads.contains(&(0, 1)));
        // Row 1 pixel (i >= 1): the reference is the pixel itself; its
        // neighbour (0, 1) shares the label but 1*5 > 5 is false, and no
        // other same-label pixel exists in the window -> also a head.
        assert!(heads.contains(&(1, 1)));
    }

    #[test]
    fn scan_nan_distance_never_suppresses_head() {
        // A NaN distance under a same-label neighbour must not count as a
        // greater value (NaN > d is false), so the center stays a head.
        let ncols = 4;
        let nrows = 40;
        let mut skeleton = vec![0.0f32; nrows * ncols];
        let mut distances = vec![0.0f32; nrows * ncols];
        for r in [2usize, 3usize] {
            skeleton[r * ncols + 1] = 1.0;
        }
        distances[2 * ncols + 1] = 2.0;
        distances[3 * ncols + 1] = f32::NAN;
        let labels = label_components(
            &skeleton.iter().map(|&v| v != 0.0).collect::<Vec<bool>>(),
            nrows,
            ncols,
        );
        let counts = vec![2i64];
        let scan = WindowScan {
            skeleton: &skeleton,
            distances: &distances,
            labels: &labels,
            counts: &counts,
            threshold: 0.0,
            nrows,
            ncols,
            search_box: 30,
        };
        let heads = scan.run(&mut |_| Ok(())).unwrap();
        assert!(heads.contains(&(2, 1)));
    }
}
