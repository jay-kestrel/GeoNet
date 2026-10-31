# SPDX-License-Identifier: GPL-3.0-or-later
module NonlinearFilter

using ..NumPyCompat: numpy_pairwise_sum, gradient_along_axes, mquantiles_p

export simple_gaussian_smoothing, anisodiff, lambda_nonlinear_filter

# Half-window NaN means always run over halfKernelWidth elements (2 for the
# kernelWidth=5 fixture); at that length numpy's pairwise order is the naive
# order, so a seed-zero naive accumulation reproduces np.nanmean exactly.
function _nanmean_window(vals::AbstractVector{Float32})::Float32
    acc = zero(Float32)
    cnt = 0
    @inbounds for v in vals
        if isnan(v)
            acc += zero(Float32)
        else
            acc += v
            cnt += 1
        end
    end
    cnt == 0 && return NaN32
    return acc / Float32(cnt)
end

"""
    simple_gaussian_smoothing(inputDemArray, kernelWidth, diffusionSigmaSquared)

Port of `pygeonet_nonlinear_filter.simple_gaussian_smoothing`. Builds a
normalized `kernelWidth`-squared 2-D Gaussian in Float64, pads the DEM with
per-row/per-column NaN-mean strips exactly as the Python does (horizontal
strips first; corner cells therefore inherit the horizontal means; the
vertical strips stack each mean twice), then runs a 'valid'-mode Float64
direct convolution.

Summation order (documented per the equivalence gate): the convolution
accumulates kernel elements in row-major (ii, then jj) order. scipy's C
loop order may differ, but both are Float64 sums whose disagreement is
orders of magnitude below the 1e-6 relative gate; the Gaussian is
symmetric, so scipy's kernel flip is immaterial.

The Python's `fillvalue = np.nanmean(...)` (pygeonet_nonlinear_filter.py:45)
is dead code and is not ported.
"""
function simple_gaussian_smoothing(inputDemArray::AbstractMatrix{Float32},
                                   kernelWidth::Integer,
                                   diffusionSigmaSquared::Real)
    Ny, Nx = size(inputDemArray)
    halfKernelWidth = Int((kernelWidth - 1) ÷ 2)
    x = collect(range(-halfKernelWidth, halfKernelWidth; length=kernelWidth))
    xv = [x[j] for i in 1:kernelWidth, j in 1:kernelWidth]
    yv = [x[i] for i in 1:kernelWidth, j in 1:kernelWidth]
    gaussianFilter = exp.(.-(xv .^ 2 .+ yv .^ 2) ./ (2 * Float64(diffusionSigmaSquared)))
    gaussianFilter ./= numpy_pairwise_sum(vec(gaussianFilter))

    hkw = halfKernelWidth
    xL = [_nanmean_window(view(inputDemArray, i, 1:hkw)) for i in 1:Ny]
    xR = [_nanmean_window(view(inputDemArray, i, Nx-hkw+1:Nx)) for i in 1:Ny]
    part1 = hcat(xL, xL)
    part2 = hcat(xR, xR)
    eI = hcat(part1, inputDemArray, part2)
    xU = [_nanmean_window(view(eI, 1:hkw, j)) for j in 1:Nx+2*hkw]
    xD = [_nanmean_window(view(eI, Ny-hkw+1:Ny, j)) for j in 1:Nx+2*hkw]
    part3 = vcat(xU', xU')
    part4 = vcat(xD', xD')
    eI = vcat(part3, eI, part4)

    smoothedDemArray = Matrix{Float64}(undef, Ny, Nx)
    @inbounds for j in 1:Nx, i in 1:Ny
        acc = 0.0
        for jj in 1:kernelWidth, ii in 1:kernelWidth
            acc += Float64(eI[i + ii - 1, j + jj - 1]) * gaussianFilter[ii, jj]
        end
        smoothedDemArray[i, j] = acc
    end
    return smoothedDemArray
end

"""
    anisodiff(img, niter, kappa, gamma, step=(1f0, 1f0), option=2)

Port of `pygeonet_nonlinear_filter.anisodiff` (Perona-Malik anisotropic
diffusion), bug for bug:

- the image is cast to Float32 up front and every operation stays Float32
  (NumPy's weak-scalar promotion keeps Python floats in the array dtype);
- `deltaS`/`deltaE` are zero-initialized and their last row/column is never
  written inside the loop, exactly like `np.diff` assignments into
  `deltaS[:-1, :]` / `deltaE[:, :-1]`;
- option 2 is the rational conductance `1/(1+(d/kappa)^2)/step` (the
  pipeline default, PeronaMalik2), option 1 the exponential one;
- NaN cells: conductance terms become NaN, the shifted North/West
  subtraction produces NaN, NaNs are zeroed, and only cells that were NaN
  in BOTH the N/S and E/W updates are re-masked to NaN before the
  `imgout += gamma*NS` update;
- `step1`/`step2` locals in the Python are dead and are not ported.

The update uses explicit temporaries (`gNS`, `E`, `S`) so every multiply
and add rounds separately, matching NumPy's temporary-array evaluation
order rather than relying on broadcast fusion.
"""
function anisodiff(img::AbstractMatrix{<:Real}, niter::Integer, kappa::Real,
                   gamma::Real, step::Tuple{Real,Real}=(1f0, 1f0), option::Integer=2)
    imgf = Float32.(img)
    imgout = copy(imgf)
    Ny, Nx = size(imgf)
    deltaS = zeros(Float32, Ny, Nx)
    deltaE = zeros(Float32, Ny, Nx)
    NS = zeros(Float32, Ny, Nx)
    EW = zeros(Float32, Ny, Nx)
    gS = ones(Float32, Ny, Nx)
    gE = ones(Float32, Ny, Nx)
    k32 = Float32(kappa)
    s0 = Float32(step[1])
    s1 = Float32(step[2])
    g32 = Float32(gamma)
    for _ in 1:niter
        @inbounds begin
            deltaS[1:Ny-1, :] .= view(imgout, 2:Ny, :) .- view(imgout, 1:Ny-1, :)
            deltaE[:, 1:Nx-1] .= view(imgout, :, 2:Nx) .- view(imgout, :, 1:Nx-1)
        end
        if option == 2
            gS .= 1f0 ./ (1f0 .+ (deltaS ./ k32) .^ 2f0) ./ s0
            gE .= 1f0 ./ (1f0 .+ (deltaE ./ k32) .^ 2f0) ./ s1
        elseif option == 1
            tS = (deltaS ./ k32) .^ 2f0
            tE = (deltaE ./ k32) .^ 2f0
            gS .= exp.(.-tS) ./ s0
            gE .= exp.(.-tE) ./ s1
        else
            throw(ArgumentError("option must be 1 or 2; got $option"))
        end
        @inbounds begin
            E = gE .* deltaE
            S = gS .* deltaS
            NS .= S
            EW .= E
            NS[2:Ny, :] .-= view(S, 1:Ny-1, :)
            EW[:, 2:Nx] .-= view(E, :, 1:Nx-1)
            mNS = isnan.(NS)
            mEW = isnan.(EW)
            NS[mNS] .= 0f0
            EW[mEW] .= 0f0
            NS .+= EW
            mNS .&= mEW
            NS[mNS] .= NaN32
            gNS = g32 .* NS
            imgout .+= gNS
        end
    end
    return imgout
end

"""
    lambda_nonlinear_filter(nanDemArray, demPixelScale) -> Float64

Port of `pygeonet_nonlinear_filter.lambda_nonlinear_filter`: the 90th-
quantile edge threshold lambda for the Perona-Malik conductance. Computes
the slope magnitude with the exact `np.gradient` scheme (Float32), strips
NaNs, and takes `scipy mquantiles` (Cunnane plotting positions, default
alphap=betap=0.4) of the absolute magnitudes at
`pygeonet_defaults.demSmoothingQuantile` = 0.9.

The pixel scale is passed explicitly here; in the Python it is read from
the module-global `Parameters.demPixelScale`.
"""
function lambda_nonlinear_filter(nanDemArray::AbstractMatrix{Float32},
                                 demPixelScale::Real)
    slopeYArray, slopeXArray = gradient_along_axes(nanDemArray, demPixelScale)
    slopeMagnitudeDemArray = sqrt.(slopeXArray .^ 2 .+ slopeYArray .^ 2)
    flat = vec(slopeMagnitudeDemArray)   # order is irrelevant below: NaN-strip, abs, sort
    valid = flat[.!isnan.(flat)]
    return mquantiles_p(abs.(valid), 0.9)
end

end # module NonlinearFilter
