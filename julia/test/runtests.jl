# SPDX-License-Identifier: GPL-3.0-or-later
# Equivalence gate for the hand-rolled fast-marching solver: replay the
# committed skfmm travel-time goldens (see fixtures/generate_fmm_goldens.py)
# through GeoNetPorts.FastMarching.travel_time and require <= 1e-6 relative
# agreement on every unmasked cell, per the build spec's equivalence bar.

using Test
using GeoNetPorts
using NPZ
using SHA

const FIXTURES = joinpath(@__DIR__, "fixtures")
const RTOL = 1.0e-6

sha256_file(path) = SHA.bytes2hex(SHA.sha256(read(path)))

function load_params(case_dir)
    dxs = Float64[]
    order = 0
    for line in eachline(joinpath(case_dir, "params.txt"))
        isempty(strip(line)) && continue
        key, value = split(strip(line), "=")
        if key == "dx"
            append!(dxs, parse.(Float64, split(value, ",")))
        elseif key == "order"
            order = parse(Int, value)
        end
    end
    dx = length(dxs) == 1 ? dxs[1] : Tuple(dxs)
    return (dx=dx, order=order)
end

function case_dirs()
    cases = filter(isdir, readdir(FIXTURES; join=true))
    return sort(cases)
end

@testset "GeoNetPorts" begin
    @testset "fixture integrity (sha256)" begin
        for line in eachline(joinpath(FIXTURES, "manifest.sha256"))
            isempty(strip(line)) && continue
            digest, rel = split(strip(line), "  ")
            @test sha256_file(joinpath(FIXTURES, rel)) == digest
        end
    end

    @testset "skfmm golden equivalence (rtol ≤ $RTOL)" begin
        @test length(case_dirs()) >= 7
        for case_dir in case_dirs()
            name = basename(case_dir)
            params = load_params(case_dir)
            phi = NPZ.npzread(joinpath(case_dir, "phi.npy"))
            speed = NPZ.npzread(joinpath(case_dir, "speed.npy"))
            ref = NPZ.npzread(joinpath(case_dir, "travel_time.npy"))
            refmask = NPZ.npzread(joinpath(case_dir, "mask.npy"))

            t = travel_time(phi, speed, params.dx; order=params.order)

            @testset "$name" begin
                # The port returns NaN where skfmm masks: same cell sets.
                @test findall(isnan, t) == findall(refmask)
                err = 0.0
                for i in eachindex(ref)
                    refmask[i] && continue
                    err = max(err, abs(t[i] - ref[i]) / max(abs(ref[i]), 1.0e-300))
                end
                @test err <= RTOL
                println("$name: max relative error = $err")
            end
        end
    end

    @testset "hand-checked scheme values" begin
        # skfmm's documented 1-D behavior: interface between the phi=-1 and
        # phi=+1 cells; marching both directions. Mirrors the pinned probe.
        phi = [-1.0 1.0 1.0 1.0]
        t = travel_time(phi, ones(1, 4), 1.0)
        @test all(isapprox.(t, [0.5 0.5 1.5 2.5]; rtol=1.0e-12))
    end

    @testset "determinism (repeated solves bit-identical)" begin
        case_dir = joinpath(FIXTURES, "smooth_hills")
        params = load_params(case_dir)
        phi = NPZ.npzread(joinpath(case_dir, "phi.npy"))
        speed = NPZ.npzread(joinpath(case_dir, "speed.npy"))
        t1 = travel_time(phi, speed, params.dx; order=params.order)
        t2 = travel_time(phi, speed, params.dx; order=params.order)
        @test isequal(t1, t2)
    end

    @testset "input validation and domain errors" begin
        @test_throws ArgumentError travel_time(ones(4, 4), ones(2, 4), 1.0)
        @test_throws ArgumentError travel_time(ones(4, 4), ones(4, 4), 0.0)
        @test_throws ArgumentError travel_time(ones(4, 4), ones(4, 4), (1.0, 0.0))
        @test_throws ArgumentError travel_time(ones(4, 4), ones(4, 4), 1.0; order=3)
        # All-positive phi: no zero contour -> nothing to march from.
        @test_throws FastMarchingError travel_time(ones(4, 4), ones(4, 4), 1.0)
        # All-zero speed: everything masked, no frozen cells.
        @test_throws FastMarchingError travel_time(fill(-1.0, 4, 4), zeros(4, 4), 1.0)
        # Float32 inputs are accepted and widened.
        t = travel_time(Float32.([-1 1 1 1]), ones(Float32, 1, 4), 1.0)
        @test all(isapprox.(t, [0.5 0.5 1.5 2.5]; rtol=1.0e-5))
    end
end
