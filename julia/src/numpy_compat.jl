# SPDX-License-Identifier: GPL-3.0-or-later
module NumPyCompat

export numpy_pairwise_sum, gradient_along_axes, nanmean, nanstd, mquantiles_p

"""
    numpy_pairwise_sum(a::AbstractVector{<:Union{Float32,Float64}}) -> T

NumPy's exact pairwise summation (numpy/core/src/umath/loops.c.src,
`pair_sum`): seed-zero naive accumulation below 8 elements, eight blocked
accumulators combined two-by-two up to 128, and divide-and-conquer splits
aligned to multiples of 8 beyond that. Reproducing this exact order makes
Float32/Float64 reductions agree with NumPy bit for bit; a naive Julia
`sum` drifts badly on Float32 (measured: 7.3e3 vs 7.97e-1 on a 9216-cell
N(0,1) slope array).
"""
function numpy_pairwise_sum(a::AbstractVector{T})::T where {T<:Union{Float32,Float64}}
    n = length(a)
    n == 0 && return zero(T)
    if n < 8
        res = zero(T)
        @inbounds for i in 1:n
            res += a[i]
        end
        return res
    elseif n <= 128
        local res::T
        @inbounds begin
            r0 = a[1]; r1 = a[2]; r2 = a[3]; r3 = a[4]
            r4 = a[5]; r5 = a[6]; r6 = a[7]; r7 = a[8]
            i = 9
            last = n - (n % 8)
            while i <= last
                r0 += a[i];     r1 += a[i + 1]; r2 += a[i + 2]; r3 += a[i + 3]
                r4 += a[i + 4]; r5 += a[i + 5]; r6 += a[i + 6]; r7 += a[i + 7]
                i += 8
            end
            res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while i <= n
                res += a[i]
                i += 1
            end
        end
        return res
    else
        n2 = n ÷ 2
        n2 -= n2 % 8
        return numpy_pairwise_sum(view(a, 1:n2)) +
               numpy_pairwise_sum(view(a, n2 + 1:n))
    end
end

"""
    gradient_along_axes(f::AbstractMatrix{Float32}, h::Real) -> (g0, g1)

`np.gradient(f, h)` for a 2-D array with a scalar spacing: second-order
central differences in the interior, first-order one-sided differences at
the edges, zeros along any length-1 axis. All arithmetic in Float32
(NumPy 2.x promotes weak Python-float scalars to the array dtype), with
the interior divisor computed as `Float32(2.0 * h)` exactly as NumPy's
`(2. * dx)` scalar is cast for a Float32 array. Returns gradients along
axis 0 and axis 1 respectively (NumPy's return order).
"""
function gradient_along_axes(f::AbstractMatrix{Float32}, h::Real)
    h64 = Float64(h)
    h32 = Float32(h64)
    h2_32 = Float32(2.0 * h64)
    ny, nx = size(f)
    g0 = Matrix{Float32}(undef, ny, nx)
    g1 = Matrix{Float32}(undef, ny, nx)
    @inbounds begin
        for j in 1:nx
            if ny == 1
                g0[1, j] = zero(Float32)
            else
                g0[1, j] = (f[2, j] - f[1, j]) / h32
                g0[ny, j] = (f[ny, j] - f[ny - 1, j]) / h32
                for i in 2:ny-1
                    g0[i, j] = (f[i + 1, j] - f[i - 1, j]) / h2_32
                end
            end
        end
        for i in 1:ny
            if nx == 1
                g1[i, 1] = zero(Float32)
            else
                g1[i, 1] = (f[i, 2] - f[i, 1]) / h32
                g1[i, nx] = (f[i, nx] - f[i, nx - 1]) / h32
                for j in 2:nx-1
                    g1[i, j] = (f[i, j + 1] - f[i, j - 1]) / h2_32
                end
            end
        end
    end
    return g0, g1
end

"""
    nanmean(x::AbstractVector{Float32}) -> Float32

`np.nanmean`: NaN cells are zero-filled before the (pairwise) sum, the
count only includes non-NaN cells, and the division happens in Float32.
An all-NaN input yields NaN (NumPy's empty-slice warning becomes NaN).
"""
function nanmean(x::AbstractVector{Float32})
    z = Vector{Float32}(undef, length(x))
    cnt = 0
    @inbounds for i in eachindex(x)
        v = x[i]
        if isnan(v)
            z[i] = zero(Float32)
        else
            z[i] = v
            cnt += 1
        end
    end
    cnt == 0 && return NaN32
    return numpy_pairwise_sum(z) / Float32(cnt)
end

"""
    nanstd(x::AbstractVector{Float32}) -> Float32

`np.nanstd` with the default `ddof=0` for Float32 input: mean and squared
deviations accumulate in Float32 (NumPy does not widen), summed pairwise,
then square-rooted. Call sites pass finite-only data, so this is NumPy's
plain-variance path.
"""
function nanstd(x::AbstractVector{Float32})
    z = Vector{Float32}(undef, length(x))
    cnt = 0
    @inbounds for i in eachindex(x)
        v = x[i]
        if isnan(v)
            z[i] = zero(Float32)
        else
            z[i] = v
            cnt += 1
        end
    end
    cnt == 0 && return NaN32
    mu = numpy_pairwise_sum(z) / Float32(cnt)
    d2 = Vector{Float32}(undef, length(x))
    @inbounds for i in eachindex(z)
        d = z[i] - mu
        d2[i] = d * d
    end
    return sqrt(numpy_pairwise_sum(d2) / Float32(cnt))
end

"""
    mquantiles_p(x, p; alphap=0.4, betap=0.4) -> Float64

`scipy.stats.mstats.mquantiles` on a 1-D array (defaults alphap=betap=0.4,
the Cunnane plotting positions — NOT the type-7 linear interpolation that
`np.percentile`/`Julia quantile` use). Operates on already-NaN-stripped
data, matching the only GeoNet call site. Q(p) = (1-gamma)*x[k] +
gamma*x[k+1] over the sorted sample, where aleph = n*p + m with
m = alphap + p*(1 - alphap - betap), k = floor(clamp(aleph, 1, n-1)), and
gamma = clamp(aleph - k, 0, 1).
"""
function mquantiles_p(x::AbstractVector{Float32}, p::Real;
                      alphap::Float64=0.4, betap::Float64=0.4)
    xs = sort(x)
    n = length(xs)
    n == 0 && throw(ArgumentError("mquantiles of an empty sample is masked in scipy"))
    n == 1 && return Float64(xs[1])
    pf = Float64(p)
    m = alphap + pf * (1.0 - alphap - betap)
    aleph = Float64(n) * pf + m
    k = floor(Int, clamp(aleph, 1.0, Float64(n - 1)))
    gamma = clamp(aleph - k, 0.0, 1.0)
    return (1.0 - gamma) * Float64(xs[k]) + gamma * Float64(xs[k + 1])
end

end # module NumPyCompat
