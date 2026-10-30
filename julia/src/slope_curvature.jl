# SPDX-License-Identifier: GPL-3.0-or-later
module SlopeCurvature

using ..NumPyCompat: gradient_along_axes, nanmean, nanstd

export compute_dem_slope, compute_dem_curvature

"""
    compute_dem_slope(filteredDemArray, pixelDemScale) -> Matrix{Float32}

Port of `pygeonet_slope_curvature.compute_dem_slope`: elementwise
`sqrt(slopeX^2 + slopeY^2)` on Float32 gradients with the exact np.gradient
scheme. The Python's percentile/slope-statistics prints are diagnostics
only and are not ported.
"""
function compute_dem_slope(filteredDemArray::AbstractMatrix{Float32},
                           pixelDemScale::Real)
    slopeYArray, slopeXArray = gradient_along_axes(filteredDemArray, pixelDemScale)
    slopeDemArray = sqrt.(slopeXArray .^ 2 .+ slopeYArray .^ 2)
    return slopeDemArray
end

"""
    compute_dem_curvature(demArray, pixelDemScale, curvatureCalcMethod)
        -> (curvatureDemArray, curvatureDemMean, curvatureDemStdDevn)

Port of `pygeonet_slope_curvature.compute_dem_curvature`, preserving the
axis naming of the original: `np.gradient` returns the axis-0 gradient
first, which the Python names `gradYArray`. The 'geometric' method
normalizes the gradients by the slope magnitude (0/0 -> NaN, matching the
numpy `seterr(divide='ignore', invalid='ignore')` contract at module
import), the 'laplacian' method skips normalization. The second gradient
pass keeps only the python-named `gradGradXArray` (axis-1 gradient of
gradX) and `gradGradYArray` (axis-0 gradient of gradY); NaN curvature
cells are zeroed before the statistics. Mean/std are Float32 nan-reductions
over the finite cells, returned as Float32.
"""
function compute_dem_curvature(demArray::AbstractMatrix{Float32},
                               pixelDemScale::Real,
                               curvatureCalcMethod::AbstractString)
    gradYArray, gradXArray = gradient_along_axes(demArray, pixelDemScale)
    slopeArrayT = sqrt.(gradXArray .^ 2 .+ gradYArray .^ 2)
    if curvatureCalcMethod == "geometric"
        gradXArrayT = gradXArray ./ slopeArrayT
        gradYArrayT = gradYArray ./ slopeArrayT
    elseif curvatureCalcMethod == "laplacian"
        gradXArrayT = gradXArray
        gradYArrayT = gradYArray
    else
        throw(ArgumentError("curvatureCalcMethod must be geometric or laplacian; " *
                            "got $(repr(curvatureCalcMethod))"))
    end
    _, gradGradXArray = gradient_along_axes(gradXArrayT, pixelDemScale)
    gradGradYArray, _ = gradient_along_axes(gradYArrayT, pixelDemScale)
    curvatureDemArray = gradGradXArray .+ gradGradYArray
    curvatureDemArray[isnan.(curvatureDemArray)] .= 0f0
    # Row-major iteration mirrors numpy's boolean-mask extract (`.ravel()`
    # order): the pairwise-sum reduction below is order-sensitive, and a
    # column-major `vec()` reorders the tree (~ulp-level mean drift).
    ny, nx = size(curvatureDemArray)
    finiteList = Float32[curvatureDemArray[i, j]
                         for i in 1:ny for j in 1:nx
                         if isfinite(curvatureDemArray[i, j])]
    curvatureDemMean = nanmean(finiteList)
    curvatureDemStdDevn = nanstd(finiteList)
    return curvatureDemArray, curvatureDemMean, curvatureDemStdDevn
end

end # module SlopeCurvature
