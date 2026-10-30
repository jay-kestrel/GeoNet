// SPDX-License-Identifier: GPL-3.0-or-later
use ndarray::{s, ArrayView2};
use numpy::{Element, PyArray1, PyReadonlyArray2, PyUntypedArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rayon::prelude::*;

const ROWS_PER_CHUNK: usize = 128;

trait BasinLabel: Copy + Send + Sync {
    fn index(self, labels: usize) -> Option<usize>;
}

macro_rules! integer_labels {
    ($($kind:ty),+) => {$(
        impl BasinLabel for $kind {
            fn index(self, labels: usize) -> Option<usize> {
                usize::try_from(self).ok().filter(|&n| n > 0 && n <= labels).map(|n| n - 1)
            }
        }
    )+};
}
integer_labels!(i8, i16, i32, i64, u8, u16, u32, u64);

macro_rules! float_labels {
    ($($kind:ty),+) => {$(
        impl BasinLabel for $kind {
            fn index(self, labels: usize) -> Option<usize> {
                if self >= 1.0 && self <= labels as Self && self.fract() == 0.0 {
                    (self as usize).checked_sub(1).filter(|&index| index < labels)
                } else {
                    None
                }
            }
        }
    )+};
}
float_labels!(f32, f64);

impl BasinLabel for bool {
    fn index(self, labels: usize) -> Option<usize> {
        (self && labels > 0).then_some(0)
    }
}

fn count_basins<T: BasinLabel>(basins: ArrayView2<'_, T>, labels: usize) -> Vec<usize> {
    if labels == 0 {
        return Vec::new();
    }
    // Indexed chunks and a serial merge make the result independent of the
    // Rayon thread count. Only integers are reduced; floats are never summed.
    let chunks = basins.nrows().div_ceil(ROWS_PER_CHUNK);
    let partials: Vec<Vec<usize>> = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let start = chunk * ROWS_PER_CHUNK;
            let end = (start + ROWS_PER_CHUNK).min(basins.nrows());
            let mut counts = vec![0; labels];
            for &value in basins.slice(s![start..end, ..]) {
                if let Some(index) = value.index(labels) {
                    counts[index] += 1;
                }
            }
            counts
        })
        .collect();
    let mut counts = vec![0; labels];
    for partial in partials {
        for (total, count) in counts.iter_mut().zip(partial) {
            *total += count;
        }
    }
    counts
}

fn select_basins(counts: &[usize], pixels: f64, minimum: i64, threshold: f64) -> Vec<bool> {
    counts
        .iter()
        .map(|&count| {
            // Keep the oracle's multiply-then-divide order and strict gates.
            let percent = count as f64 * 100.00001 / pixels;
            percent > threshold && (minimum < 0 || count > minimum as usize)
        })
        .collect()
}

fn typed_counts<T: BasinLabel + Element>(
    array: PyReadonlyArray2<'_, T>,
    labels: usize,
) -> Vec<usize> {
    count_basins(array.as_array(), labels)
}

/// Return the label-order qualification mask; the two outlet arguments are
/// unused by the reference kernel and intentionally accepted without coercion.
#[pyfunction]
#[allow(non_snake_case, unused_variables, clippy::too_many_arguments)]
pub fn identify_start_points(
    outlet_array: &Bound<'_, PyAny>,
    basinIndexArray: &Bound<'_, PyUntypedArray>,
    fastMarchingStartPointList: &Bound<'_, PyAny>,
    nDempixels: f64,
    basin_elements: i64,
    threshold: f64,
    iter_total: usize,
) -> PyResult<Py<PyArray1<bool>>> {
    macro_rules! dispatch {
        ($($kind:ty),+) => {$(
            if let Ok(array) = basinIndexArray.extract::<PyReadonlyArray2<'_, $kind>>() {
                let counts = typed_counts(array, iter_total);
                let keep = select_basins(&counts, nDempixels, basin_elements, threshold);
                return Ok(PyArray1::from_vec(basinIndexArray.py(), keep).unbind());
            }
        )+};
    }
    dispatch!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);
    Err(PyTypeError::new_err(
        "basinIndexArray must be a two-dimensional native-endian integer, float32, float64, or boolean NumPy array",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn empty_mask() {
        assert_eq!(count_basins(array![[0_i16]].view(), 0), Vec::<usize>::new());
        assert_eq!(select_basins(&[0], 1.0, 0, 0.0), [false]);
    }

    #[test]
    fn single_component_and_single_pixel() {
        assert_eq!(count_basins(array![[1_i16, 1], [1, 1]].view(), 1), [4]);
        assert_eq!(select_basins(&[4], 4.0, 2, 0.1), [true]);
        assert_eq!(select_basins(&[1], 1.0, 0, 0.1), [true]);
        assert_eq!(select_basins(&[1], 1.0, 1, 0.1), [false]);
    }

    #[test]
    fn strict_gates_and_non_labels() {
        let basins = array![[f64::NAN, f64::INFINITY, -1.0, 0.0, 1.5, 1.0, 2.0]];
        assert_eq!(count_basins(basins.view(), 1), [1]);
        assert_eq!(select_basins(&[1], 100.00001, 0, 1.0), [false]);
    }

    #[test]
    fn zero_area_preserves_parallel_numba_ieee_division() {
        assert_eq!(select_basins(&[1, 0], 0.0, 0, 0.0), [true, false]);
        assert_eq!(select_basins(&[1, 0], -0.0, 0, 0.0), [false, false]);
    }

    #[test]
    fn fixed_chunk_merge_with_strided_views() {
        let basins = ndarray::Array2::from_shape_fn((300, 4), |(r, c)| (r + c) % 3 + 1);
        let basins = basins.mapv(|value| value as i32);
        let view = basins.slice(s![..;-1, ..;2]);
        let expected: Vec<usize> = (1..=3)
            .map(|label| view.iter().filter(|&&value| value == label).count())
            .collect();
        for threads in [1, 2, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            assert_eq!(pool.install(|| count_basins(view, 3)), expected);
        }
    }
}
