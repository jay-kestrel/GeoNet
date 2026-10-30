# SPDX-License-Identifier: GPL-3.0-or-later
module GeoNetPorts

include("numpy_compat.jl")
include("nonlinear_filter.jl")
include("slope_curvature.jl")
include("start_points.jl")

using .NumPyCompat
using .NonlinearFilter
using .SlopeCurvature
using .StartPoints

# Julia port of GeoNet's core numeric functionality (stage-1 Perona-Malik
# chain, stage-2 slope/curvature, stage-5 start-point kernel) plus the
# NumPy/SciPy numeric-semantics shims they rely on. Structured as one
# submodule per Python stage so the follow-on fast-marching solver can land
# beside these as another include() without touching the existing kernels.
export simple_gaussian_smoothing, anisodiff, lambda_nonlinear_filter,
       compute_dem_slope, compute_dem_curvature,
       fast_marching_start_point_identification, fmm_list_creation,
       numpy_pairwise_sum, gradient_along_axes, nanmean, nanstd, mquantiles_p

end # module
