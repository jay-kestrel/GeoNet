// SPDX-License-Identifier: GPL-3.0-or-later
use numpy::{PyArray1, PyUntypedArray};
use pyo3::prelude::*;

type HeadCoordinates = (Py<PyArray1<isize>>, Py<PyArray1<isize>>);

/// Signature of Channel_Head_Definition; returns (xx, yy), never writes files.
/// Later integration must leave plotting and raster/vector writes in Python.
#[pyfunction]
#[allow(non_snake_case, unused_variables)]
pub fn find_channel_heads(
    skeletonFromFlowAndCurvatureArray: &Bound<'_, PyUntypedArray>,
    geodesicDistanceArray: &Bound<'_, PyUntypedArray>,
) -> PyResult<HeadCoordinates> {
    unimplemented!("find_channel_heads is a scaffold; use the original Python pipeline")
}
