# SPDX-License-Identifier: GPL-3.0-or-later
# Travel-time solver benchmark: warm timings of
# GeoNetPorts.FastMarching.travel_time at the pipeline's grid spacing
# (dx = 0.01). Run from the julia/ directory:
#     julia --project=. bench/fmm_bench.jl
#
# Terrain generators are self-contained (seeded xorshift perturbation, no
# package RNG) so timings are reproducible across runs.

using GeoNetPorts
using Printf

const FM = GeoNetPorts.FastMarching

function xorshift_noise(n, seed)
    out = Vector{Float64}(undef, n * n)
    state = seed % UInt64 == 0 ? UInt64(0x9E3779B97F4A7C15) : seed % UInt64
    for i in eachindex(out)
        state = state ^ (state << 13)
        state = state ^ (state >> 7)
        state = state ^ (state << 17)
        out[i] = (state >> 11) / 2^53
    end
    return out
end

function rough_speed(n, seed)
    c = n / 2.0
    speed = ones(n, n)
    for j in 1:n, i in 1:n
        speed[i, j] += 1.5 * exp(-((i - c)^2 + (j - c)^2) / (n / 4)^2)
    end
    speed .+= 0.05 .* reshape(xorshift_noise(n, seed), n, n)
    return max.(speed, 1.0)  # reciprocal-local-cost floor analog
end

"Pipeline-shaped phi: 0 outside a square basin, 1 inside, -1 at the start."
function basin_phi(n)
    phi = ones(n, n)
    phi[1:18, :] .= 0.0
    phi[n-17:n, :] .= 0.0
    phi[:, 1:18] .= 0.0
    phi[:, n-17:n] .= 0.0
    phi[n ÷ 2, n ÷ 2] = -1.0
    return phi
end

function bench_case(label, phi, speed; repeats=3)
    best = Inf
    result = nothing
    for _ in 1:repeats
        t = @elapsed result = FM.travel_time(phi, speed, 0.01; order=2)
        best = min(best, t)
    end
    @printf("%-28s %5d×%-5d %9.1f ms  (masked/NaN cells: %d)\n",
            label, size(phi, 1), size(phi, 2), best * 1000, count(isnan, result))
    return best
end

function main()
    println("GeoNetPorts.FastMarching.travel_time — order 2, dx=0.01, warm, best of 3")
    println("=======================================================================")
    for n in (256, 512, 1024)
        phi = basin_phi(n)
        speed = rough_speed(n, UInt64(n))
        bench_case("basin, constant speed", phi, ones(n, n))
        bench_case("basin, rough speed", phi, speed)
        full = ones(n, n)
        full[n ÷ 2, n ÷ 2] = -1.0
        bench_case("full grid, rough speed", full, speed)
    end
end

main()
