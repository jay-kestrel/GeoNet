# SPDX-License-Identifier: GPL-3.0-or-later
# GeoNetPorts — Julia ports of GeoNet's core numeric functionality.
#
# One submodule per Python stage, mirroring the pygeonet_* module names.
# The subproject is self-contained: kernels read plain arrays / .npy fixtures
# and never import the Python modules.

module GeoNetPorts

include("numpy_compat.jl")
include("nonlinear_filter.jl")
include("slope_curvature.jl")
include("start_points.jl")
include("FastMarching.jl")

using .NumPyCompat
using .NonlinearFilter
using .SlopeCurvature
using .StartPoints
using .FastMarching

export FastMarching, travel_time, FastMarchingError,
       simple_gaussian_smoothing, anisodiff, lambda_nonlinear_filter,
       compute_dem_slope, compute_dem_curvature,
       fast_marching_start_point_identification, fmm_list_creation,
       numpy_pairwise_sum, gradient_along_axes, nanmean, nanstd, mquantiles_p

end # module
