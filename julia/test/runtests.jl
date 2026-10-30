# SPDX-License-Identifier: GPL-3.0-or-later
#
# Equivalence suite for the GeoNetPorts kernels against seeded synthetic
# goldens produced by julia/tools/generate_goldens.py on the reference
# stack (python 3.13.14 / numpy 2.3.5 / scipy 1.17.1 / numba 0.62.1 — see
# goldens/manifest.json). Gates:
#   * bit-exact (raw Float32/integer bytes) for elementwise kernels;
#   * <=1e-6 relative for the scipy convolve2d (Float64 convolution,
#     summation order documented in nonlinear_filter.jl).
# Oracle for the start-point kernel: the pure-NumPy reference ported from
# tests/test_stage5_determinism.py:reference_keep.

using Test
using NPZ
using SHA
using GeoNetPorts

const GOLDENS = joinpath(@__DIR__, "goldens")

load(name) = npzread(joinpath(GOLDENS, name))

const MANIFEST_TEXT = read(joinpath(GOLDENS, "manifest.json"), String)

# Guards against corrupted committed fixtures: every .npy is checked against
# the sha256 recorded by the generator (machine-generated JSON: sorted keys,
# indent=2, one "name": "hash" pair per file). Filenames are
# [a-z0-9_.]+ so a literal-dot replace is the only escaping needed.
function golden_sha(name::AbstractString)
    # name is [a-z0-9_]+; the only regex-special char is the dot in ".npy"
    pattern = name * "\\.npy\"\\s*:\\s*\"([0-9a-f]{64})"
    m = match(Regex(pattern), MANIFEST_TEXT)
    m === nothing && error("manifest.json has no entry for $name")
    return m.captures[1]
end

"""Max elementwise relative error; NaNs must agree positionally. An inf
result means a NaN-pattern mismatch, not a magnitude one."""
function maxrel(a::AbstractArray{Float64}, b::AbstractArray{Float64})
    size(a) == size(b) || return Inf
    err = 0.0
    for i in eachindex(a)
        ga, gb = a[i], b[i]
        if isnan(ga) || isnan(gb)
            isnan(ga) == isnan(gb) || return Inf
            continue
        end
        err = max(err, abs(ga - gb) / max(abs(gb), 1e-12))
    end
    return err
end

"""Raw-byte equality for Float32 arrays (catches NaN placement and any
payload drift, not just magnitudes)."""
bitexact_f32(a, b) = size(a) == size(b) &&
                     reinterpret(UInt8, vec(Float32.(a))) ==
                     reinterpret(UInt8, vec(Float32.(b)))

# --- oracle (ported from tests/test_stage5_determinism.py) ------------------

const BASIN_ELEMENTS = 2   # pygeonet_prepare.numBasinsElements
const THRESHOLD = 0.1      # pygeonet_defaults.thresholdPercentAreaForDelineation

"""
Port of the pure-NumPy oracle `reference_keep`
(tests/test_stage5_determinism.py:149): serial per-label count and
label-order keep mask with the production gate constants.
"""
function reference_keep(basin::AbstractMatrix{<:Real}, splist::AbstractMatrix{<:Integer};
                        basin_elements=BASIN_ELEMENTS, threshold=THRESHOLD)
    iter_total = size(splist, 2)
    keep = falses(iter_total)
    nDempixels = Float64(length(basin))
    # Python's 0-based `label` looks up basin value label+1; Julia's 1-based
    # label already IS that basin value.
    for label in 1:iter_total
        n = count(==(label), basin)
        if (n * 100.00001 / nDempixels > threshold) && (n > basin_elements)
            keep[label] = true
        end
    end
    return keep
end

# ---------------------------------------------------------------------------

@testset "golden fixture integrity" begin
    npys = sort(filter(endswith(".npy"), readdir(GOLDENS)))
    @test length(npys) == 25
    for name in npys
        @test bytes2hex(sha256(read(joinpath(GOLDENS, name)))) ==
              golden_sha(splitext(name)[1])
    end
end

@testset "NumPyCompat: scipy mquantiles (alphap=betap=0.4)" begin
    vals = load("mq_vals.npy")::Vector{Float32}
    offsets = load("mq_offsets.npy")::Vector{Int64}
    sizes = load("mq_sizes.npy")::Vector{Int64}
    golden = load("mq_quantiles.npy")::Vector{Float64}
    @test length(offsets) == length(sizes) + 1
    errs = Float64[]
    for (idx, n) in enumerate(sizes)
        seg = vals[offsets[idx]+1:offsets[idx+1]]
        @test length(seg) == n
        q = mquantiles_p(abs.(seg), 0.9)
        @test q == golden[idx]   # bit-exact float64
        push!(errs, abs(q - golden[idx]))
    end
    @test length(errs) == length(sizes)
end

@testset "stage 1: Perona-Malik chain" begin
    dem = load("pm_input_dem.npy")::Matrix{Float32}
    @test size(dem) == (96, 96)
    # exact NaN/flag counts depend on the seeded draw (a -9998 flag may land
    # on an already-NaN cell); only presence is a fixture contract
    @test count(isnan, dem) > 0
    @test count(==(Float32(-9998.0)), dem) > 0

    # gaussian smoothing: scipy convolve2d, Float64 convolution gate
    smoothed = simple_gaussian_smoothing(dem, 5, 0.05)
    golden_sm = load("pm_gaussian_smoothing.npy")::Matrix{Float64}
    err_sm = maxrel(smoothed, golden_sm)
    println("gaussian smoothing  max rel err = ", err_sm)
    @test err_sm <= 1e-6

    # lambda: scipy mquantiles on the |slope| field — must be bit-exact
    lambda_value = lambda_nonlinear_filter(dem, 30.0)
    golden_lambda = load("pm_lambda.npy")[1]::Float64
    println("lambda julia=", lambda_value, "  golden=", golden_lambda)
    @test lambda_value == golden_lambda

    # anisodiff option 2 (pipeline default, PeronaMalik2): bit-exact
    filtered = anisodiff(dem, 50, lambda_value, 0.1, (30.0, 30.0), 2)
    @test bitexact_f32(filtered, load("pm_filtered.npy"))

    # anisodiff option 1 (exponential conductance): numpy libm exp vs
    # openlibm exp may differ by ulps — measured gate
    filtered1 = anisodiff(dem, 50, lambda_value, 0.1, (30.0, 30.0), 1)
    golden1 = load("pm_filtered_opt1.npy")::Matrix{Float32}
    err1 = maxrel(Float64.(filtered1), Float64.(golden1))
    println("anisodiff option 1  max rel err = ", err1)
    @test err1 <= 1e-6

    # determinism: identical reruns (the NumPy determinism contract)
    @test bitexact_f32(anisodiff(dem, 50, lambda_value, 0.1, (30.0, 30.0), 2),
                       load("pm_filtered.npy"))
end

@testset "stage 2: slope and curvature" begin
    filtered = load("pm_filtered.npy")::Matrix{Float32}

    # slope off the golden filtered dem: bit-exact
    slope = compute_dem_slope(filtered, 30.0)
    golden_slope = load("slope_output.npy")::Matrix{Float32}
    @test bitexact_f32(slope, golden_slope)

    # curvature 'geometric' + its Float32 statistics: bit-exact
    curvature, cmean, cstd = compute_dem_curvature(filtered, 30.0, "geometric")
    @test bitexact_f32(curvature, load("curvature_output.npy"))
    stats = load("curvature_stats.npy")::Vector{Float64}
    println("curvature mean julia=", Float64(cmean), "  golden=", stats[1])
    println("curvature std  julia=", Float64(cstd), "  golden=", stats[2])
    @test Float64(cmean) == stats[1]
    @test Float64(cstd) == stats[2]

    # laplacian variant: bit-exact
    curvature_lap, _, _ = compute_dem_curvature(filtered, 30.0, "laplacian")
    @test bitexact_f32(curvature_lap, load("curvature_laplacian.npy"))
end

@testset "stage 5: start-point kernel" begin
    for case in ("edge1x32", "grid36", "mixed20")
        basin = load("sp_$(case)_basin.npy")
        splist = load("sp_$(case)_splist.npy")::Matrix{Int64}
        keep_golden = load("sp_$(case)_keep.npy") .== 1
        fmm_golden = load("sp_$(case)_fmm.npy")::Matrix{Int64}
        iter_total = size(splist, 2)

        # kernel vs oracle vs golden (the oracle is the ported pure-NumPy
        # reference; the golden is the real Numba kernel's output)
        keep = fast_marching_start_point_identification(nothing, basin, splist,
                                                        Float64(length(basin)),
                                                        BASIN_ELEMENTS, THRESHOLD,
                                                        iter_total)
        ref = reference_keep(basin, splist)
        @test keep == ref
        @test keep == keep_golden

        # deterministic serial reduction: kernel == oracle == golden fmm list
        fmm = fmm_list_creation(splist, keep)
        @test fmm == fmm_golden
        @test fmm == splist[:, findall(ref)]
        @test fmm == splist[:, findall(keep_golden)]

        # determinism: repeated runs identical regardless of thread schedule
        keep2 = fast_marching_start_point_identification(nothing, basin, splist,
                                                         Float64(length(basin)),
                                                         BASIN_ELEMENTS, THRESHOLD,
                                                         iter_total)
        @test keep2 == keep
        println("case $case: iter_total=$iter_total, kept=$(count(keep))")
    end
end
