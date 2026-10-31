# SPDX-License-Identifier: GPL-3.0-or-later
module StartPoints

export fast_marching_start_point_identification, fmm_list_creation

"""
    fast_marching_start_point_identification(outlet_array, basinIndexArray,
        fastMarchingStartPointList, nDempixels, basin_elements, threshold,
        iter_total) -> BitVector

Port of the stage-5 Numba kernel
`pygeonet_fast_marching.Fast_Marching_Start_Point_Identification`
(njit(parallel=True)). Pass 1 is parallel: each label counts its own basin
pixels and writes only its own slot of the `keep` mask — the writes are
disjoint, so `Threads.@threads` preserves the race-freedom contract the
Numba `prange` version was rewritten to guarantee (the original list-append
version raced). Pass 2 (the deterministic serial reduction) lives in
`fmm_list_creation`, which gathers start points in label order.

Gate semantics preserved verbatim, including the odd constant: a basin
qualifies when `numelments * 100.00001 / nDempixels > threshold` AND
`numelments > basin_elements` (pygeonet_prepare.numBasinsElements = 2,
pygeonet_defaults.thresholdPercentAreaForDelineation = 0.1 in the pipeline;
the tests that seeded this port pin 2 / 0.1).

`outlet_array` and `fastMarchingStartPointList` are kept in the signature
for parity with the Python kernel, which also ignores them.
"""
function fast_marching_start_point_identification(outlet_array,
        basinIndexArray::AbstractMatrix{<:Real},
        fastMarchingStartPointList::AbstractMatrix{<:Integer},
        nDempixels::Real,
        basin_elements::Real,
        threshold::Real,
        iter_total::Integer)
    # One byte per element (like numpy's bool array): concurrent writes to
    # distinct labels must be distinct memory writes. A packed BitVector
    # read-modify-writes shared words and is not thread-safe here.
    keep = fill(false, iter_total)
    # Python's 0-based `label` looks up basin value label+1; Julia's 1-based
    # label already IS that basin value.
    Threads.@threads for label in 1:iter_total
        numelments = count(==(label), basinIndexArray)
        percentBasinArea = Float64(numelments) * 100.00001 / Float64(nDempixels)
        keep[label] = (percentBasinArea > threshold) && (numelments > basin_elements)
    end
    return keep
end

"""
    fmm_list_creation(fastMarchingStartPointList, qualifyingStartPoints)
        -> Matrix{Int}

Port of `pygeonet_fast_marching.fmm_list_creation`: the deterministic
serial reduction of the qualifying start points in label order. Row 0 of
the Python's result is `fastMarchingStartPointList[0, keep]` (the outlet
rows, fmmY), row 1 the outlet columns (fmmX); the result is a 2-row matrix
of Int64 (empty 2x0 when nothing qualifies).
"""
function fmm_list_creation(fastMarchingStartPointList::AbstractMatrix{<:Integer},
                           qualifyingStartPoints::AbstractVector{<:Union{Bool,Integer}})
    cols = findall(==(true), qualifyingStartPoints)
    fmmY = fastMarchingStartPointList[1, cols]
    fmmX = fastMarchingStartPointList[2, cols]
    return vcat(fmmY', fmmX')
end

end # module StartPoints
