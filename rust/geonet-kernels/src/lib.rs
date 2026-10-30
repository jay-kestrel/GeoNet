// SPDX-License-Identifier: GPL-3.0-or-later
//! Build-only scaffold: the pipeline does not call these kernels yet.

mod find_channel_heads;
mod identify_start_points;

use pyo3::prelude::*;

#[pymodule]
fn geonet_kernels(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        identify_start_points::identify_start_points,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        find_channel_heads::find_channel_heads,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_only_the_two_kernel_functions() -> PyResult<()> {
        Python::attach(|py| {
            let module = PyModule::new(py, "geonet_kernels")?;
            geonet_kernels(&module)?;
            let mut names: Vec<String> = module
                .dict()
                .keys()
                .extract::<Vec<String>>()?
                .into_iter()
                .filter(|name| !name.starts_with('_'))
                .collect();
            names.sort();
            assert_eq!(names, ["find_channel_heads", "identify_start_points"]);
            for name in names {
                assert!(module.getattr(name)?.is_callable());
            }
            Ok(())
        })
    }
}
