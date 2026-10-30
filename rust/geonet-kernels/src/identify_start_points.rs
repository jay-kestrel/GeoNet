// SPDX-License-Identifier: GPL-3.0-or-later
use numpy::{PyArray1, PyUntypedArray};
use pyo3::prelude::*;

/// Signature of Fast_Marching_Start_Point_Identification; returns its keep mask.
/// Array dtypes are intentionally not coerced at this scaffold boundary: GRASS
/// emits integer basins, while the Python oracle also accepts floating basins.
#[pyfunction]
#[allow(non_snake_case, unused_variables, clippy::too_many_arguments)]
pub fn identify_start_points(
    outlet_array: &Bound<'_, PyUntypedArray>,
    basinIndexArray: &Bound<'_, PyUntypedArray>,
    fastMarchingStartPointList: &Bound<'_, PyUntypedArray>,
    nDempixels: f64,
    basin_elements: i64,
    threshold: f64,
    iter_total: usize,
) -> PyResult<Py<PyArray1<bool>>> {
    unimplemented!("identify_start_points is a scaffold; use the original Python pipeline")
}
