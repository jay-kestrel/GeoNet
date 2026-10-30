# SPDX-License-Identifier: GPL-3.0-or-later
# Hand-rolled 2-D fast-marching (eikonal) solver.
#
# A faithful Julia transcription of the scikit-fmm travel-time marcher that
# the GeoNet pipeline calls (skfmm.travel_time with order=2), pinned against
# the scikit-fmm 2025.6.23 sources: base_marcher.{h,cpp}, distance_marcher.cpp,
# travel_time_marcher.{h,cpp}, heap.{h,cpp}, pfmm.py. The scheme, the update
# arithmetic (including evaluation order), the initialization rule, the heap
# mechanics and the tie handling are reproduced so travel-time fields match
# skfmm within tight floating-point tolerances, not merely qualitatively.

module FastMarching

export travel_time, FastMarchingError

# base_marcher.h cell states
const FAR = 0x00
const NARROW = 0x01
const FROZEN = 0x02
const MASK = 0x03

# base_marcher.h constants
const DOUBLE_EPSILON = 2.220446049250313e-16   # numeric_limits<double>::epsilon()
const MAX_DOUBLE = 1.7976931348623157e308      # numeric_limits<double>::max()

# travel_time_marcher.cpp stencil constants
const _AA = 9.0 / 4.0
const _ONE_THIRD = 1.0 / 3.0

"""
    FastMarchingError <: Exception

Domain failure of the eikonal solve, mirroring the conditions scikit-fmm
reports as Python exceptions:

- "the array phi contains no zero contour (no zero level set)"
  (skfmm: `ValueError`);
- "Unreachable voxel" — a cell's update quadratic has no real solution and no
  reduced-dimension fallback exists (skfmm: `RuntimeError`, the condition the
  GeoNet pipeline's `travel_time_with_domain_guard` re-wraps as
  `FastMarchingDomainError`).
"""
struct FastMarchingError <: Exception
    msg::String
end

# Internal control-flow sentinel mirroring the C++ throw of
# "Negative discriminant in time marcher quadratic." inside
# travelTimeMarcher::solveQuadratic, caught in updatePointOrderTwo.
struct _NegativeDiscriminant <: Exception end

"""
    travel_time(phi, speed, dx=1.0; order=2) -> Matrix{Float64}

Travel time from the zero contour of `phi` under propagation speed `speed`,
semantics of `skfmm.travel_time` (scikit-fmm, order-2 default):

- the interface is the zero contour of `phi`; marching proceeds in *both*
  directions from the interface, so the cell holding `phi == -1` receives a
  nonzero time (the GeoNet `Fast_Marching` caller relies on this);
- cells bordering a sign change of `phi` are frozen at initialization with
  the interpolated crossing distance `d = dx·phiᵢ/(phiᵢ − φₙ)` per dimension,
  combined as `sqrt(1/Σ 1/dᵢ²)`, then divided by the cell's speed;
- cells with `speed < double-epsilon` are walls: never updated, returned as
  `NaN` here (skfmm returns them masked);
- cells still unreachable when the march ends are returned as `NaN`
  (skfmm masks the `DBL_MAX` sentinel);
- masked-array inputs, `narrow` banding and periodic boundaries are not
  supported (none are used by the GeoNet pipeline).

`dx` is the uniform grid spacing, or a `(drow, dcol)` tuple for anisotropic
grids (per-dimension order matches scikit-fmm: dimension 0 is the row axis).
`order` is the stencil order, 1 or 2 (default 2 as scikit-fmm).

Throws `FastMarchingError` when the domain admits no zero contour or a cell
cannot be reached.
"""
function travel_time(phi::AbstractMatrix{<:Real}, speed::AbstractMatrix{<:Real},
                     dx::Real=1.0; order::Integer=2)
    return travel_time(phi, speed, (Float64(dx), Float64(dx)); order=order)
end

function travel_time(phi::AbstractMatrix{<:Real}, speed::AbstractMatrix{<:Real},
                     dx::NTuple{2,Real}; order::Integer=2)
    size(phi) == size(speed) || throw(ArgumentError("phi and speed must have the same shape"))
    all(>(0), dx) || throw(ArgumentError("dx must be greater than zero"))
    order in (1, 2) || throw(ArgumentError("order must be 1 or 2"))
    return _travel_time(Matrix{Float64}(phi), Matrix{Float64}(speed),
                        Float64(dx[1]), Float64(dx[2]), Int(order))
end

mutable struct _Solver
    phi_flat::Vector{Float64}   # row-major flattening (numpy C order)
    speed_flat::Vector{Float64}
    dist::Vector{Float64}    # travel time per flat cell
    flag::Vector{UInt8}
    heapptr::Vector{Int}     # flat cell -> heap list entry (0 until pushed)
    dxrow::Float64           # cell width along the row axis (numpy dim 0)
    dxcol::Float64           # cell width along the column axis (numpy dim 1)
    idx2row::Float64         # 1/dx² along the row axis (numpy dim 0)
    idx2col::Float64         # 1/dx² along the column axis (numpy dim 1)
    nrows::Int
    ncols::Int
    order::Int
end

# base_marcher.h _getN: neighbor `dir` steps along `dim`; 0 when out of
# bounds or when the neighbor's flag equals `excluded` (skfmm's -1).
@inline function _neighbor(s::_Solver, flat::Int, dim::Int, dir::Int, excluded::UInt8)
    if dim == 1
        row = (flat - 1) ÷ s.ncols + 1
        r2 = row + dir
        (r2 < 1 || r2 > s.nrows) && return 0
        n = flat + dir * s.ncols
    else
        col = (flat - 1) % s.ncols + 1
        c2 = col + dir
        (c2 < 1 || c2 > s.ncols) && return 0
        n = flat + dir
    end
    s.flag[n] == excluded && return 0
    return n
end

# heap.{h,cpp}: indexed binary min-heap with skfmm's exact sift mechanics.
# List entries (distance/address/backpointer) are append-only; the heap array
# holds list indices. Strict `<` comparisons keep tie order deterministic.
mutable struct _Heap
    distance::Vector{Float64}
    address::Vector{Int}
    backpointer::Vector{Int}
    heap::Vector{Int}
    heaplength::Int
end

_Heap() = _Heap(Float64[], Int[], Int[], Int[], 0)

@inline _isempty(h::_Heap) = h.heaplength == 0
@inline _peek(h::_Heap) = h.distance[h.heap[1]]

function _heap_push!(h::_Heap, addr::Int, value::Float64)
    push!(h.distance, value)
    push!(h.address, addr)
    li = length(h.distance)
    h.heaplength += 1
    push!(h.backpointer, h.heaplength)
    if length(h.heap) >= h.heaplength
        h.heap[h.heaplength] = li
    else
        push!(h.heap, li)
    end
    _sift_toward_root!(h, 1, h.heaplength)
    return li
end

function _heap_pop!(h::_Heap)
    li = h.heap[1]
    value = h.distance[li]
    addr = h.address[li]
    last = h.heap[h.heaplength]
    h.heap[1] = last
    h.backpointer[last] = 1
    h.heaplength -= 1
    _sift_toward_leaves!(h, 1)
    return addr, value
end

function _heap_set!(h::_Heap, li::Int, value::Float64)
    old = h.distance[li]
    pos = h.backpointer[li]
    h.distance[li] = value
    if value > old
        _sift_toward_leaves!(h, pos)
    end
    h.distance[h.heap[pos]] == value || return
    _sift_toward_root!(h, 1, pos)
    return
end

# heap.cpp _siftDown(startPos, pos): bubble the item at `pos` up toward the
# root while smaller than its parent (naming follows skfmm, direction does not).
function _sift_toward_root!(h::_Heap, startpos::Int, pos::Int)
    d = h.distance
    newitem = h.heap[pos]
    while pos > startpos
        parentpos = pos >> 1
        parent = h.heap[parentpos]
        if d[newitem] < d[parent]
            h.heap[pos] = parent
            h.backpointer[parent] = pos
            pos = parentpos
            continue
        end
        break
    end
    h.heap[pos] = newitem
    h.backpointer[newitem] = pos
    return
end

# heap.cpp _siftUp(pos): bubble the item at `pos` down to the leaves, picking
# the smaller child (right child on ties), then let the item settle upward.
function _sift_toward_leaves!(h::_Heap, pos::Int)
    d = h.distance
    endpos = h.heaplength
    startpos = pos
    newitem = h.heap[pos]
    childpos = 2 * pos
    while childpos <= endpos
        rightpos = childpos + 1
        if rightpos <= endpos && !(d[h.heap[childpos]] < d[h.heap[rightpos]])
            childpos = rightpos
        end
        h.heap[pos] = h.heap[childpos]
        h.backpointer[h.heap[childpos]] = pos
        pos = childpos
        childpos = 2 * pos
    end
    h.heap[pos] = newitem
    _sift_toward_root!(h, startpos, pos)
    return
end

# distanceMarcher::solveQuadratic travel-time override: only the positive
# root is taken; a negative discriminant is a control-flow signal.
@inline function _solve_quadratic(s::_Solver, i::Int, a::Float64, b::Float64, c::Float64)
    c -= 1 / s.speed_flat[i]^2
    det = b^2 - 4 * a * c
    if det >= 0
        return (-b + sqrt(det)) / 2.0 / a
    end
    throw(_NegativeDiscriminant())
end

# distanceMarcher::updatePointOrderOne, with the travel-time quadratic.
# Used when order=1; the travel-time marcher never falls back to it from the
# order-2 update (its fallback is the reduced-dimension recursion below).
function _update_order_one(s::_Solver, i::Int)
    a = b = c = 0.0
    for dim in (1, 2)
        value = MAX_DOUBLE
        for dir in (-1, 1)
            n = _neighbor(s, i, dim, dir, MASK)
            if n != 0 && s.flag[n] == FROZEN && abs(s.dist[n]) < abs(value)
                value = s.dist[n]
            end
        end
        value < MAX_DOUBLE || continue
        idx2 = dim == 1 ? s.idx2row : s.idx2col
        a += idx2
        b -= idx2 * 2 * value
        c += idx2 * value^2
    end
    try
        return _solve_quadratic(s, i, a, b, c)
    catch e
        e isa _NegativeDiscriminant || rethrow()
        # C++ fallback: position of the minimum of the quadratic.
        return -b / (2.0 * a)
    end
end

# travelTimeMarcher::updatePointOrderTwo(i, avoid_dim): second-order update
# with the (4·v1 − v2)/3 extrapolation; `avoid` is a bitset of dimensions
# excluded from the solve (the negative-discriminant retry path).
_dim_bit(dim::Int)::UInt8 = UInt8(1) << (dim - 1)

function _update_order_two(s::_Solver, i::Int, avoid::UInt8)
    a = b = c = 0.0
    for dim in (1, 2)
        avoid & _dim_bit(dim) != 0 && continue
        value1 = MAX_DOUBLE
        value2 = MAX_DOUBLE
        for dir in (-1, 1)
            n = _neighbor(s, i, dim, dir, MASK)
            n == 0 && continue
            s.flag[n] == FROZEN || continue
            abs(s.dist[n]) < abs(value1) || continue
            value1 = s.dist[n]
            n2 = _neighbor(s, i, dim, 2 * dir, MASK)
            if n2 != 0 && s.flag[n2] == FROZEN &&
               ((s.dist[n2] <= value1 && value1 >= 0) ||
                (s.dist[n2] >= value1 && value1 <= 0))
                value2 = s.dist[n2]
                # A stencil spanning the interface flips the far value's sign.
                if s.phi_flat[n2] * s.phi_flat[n] < 0 || s.phi_flat[n2] * s.phi_flat[i] < 0
                    value2 = -value2
                end
            end
        end
        idx2 = dim == 1 ? s.idx2row : s.idx2col
        if value2 < MAX_DOUBLE
            tp = _ONE_THIRD * (4 * value1 - value2)
            a += idx2 * _AA
            b -= idx2 * 2 * _AA * tp
            c += idx2 * _AA * tp^2
        elseif value1 < MAX_DOUBLE
            a += idx2
            b -= idx2 * 2 * value1
            c += idx2 * value1^2
        end
    end
    try
        return _solve_quadratic(s, i, a, b, c)
    catch e
        e isa _NegativeDiscriminant || rethrow()
        # Negative discriminant: retry excluding one dimension at a time and
        # keep the smallest reduced solution (travel_time_marcher.cpp).
        if avoid == 0b11
            return Inf
        end
        best = Inf
        for dim in (1, 2)
            avoid & _dim_bit(dim) != 0 && continue
            sol = _update_order_two(s, i, avoid | _dim_bit(dim))
            best = min(best, sol)
        end
        return best
    end
end

# Public entry of the order-2 update: Inf after retries means the cell is
# unreachable — skfmm throws "Unreachable voxel" (a RuntimeError, which the
# GeoNet pipeline's domain guard re-wraps).
@inline function _update_order_two_public(s::_Solver, i::Int)
    res = _update_order_two(s, i, 0x00)
    res == Inf && throw(FastMarchingError("Unreachable voxel"))
    return res
end

# distanceMarcher::initalizeFrozen, plus the travel-time conversion
# distance -> time by division with the cell's own speed.
function _initialize_frozen!(s::_Solver)
    n = length(s.flag)
    # Zero-contour cells are exact sources.
    for i in 1:n
        if s.flag[i] != MASK && s.phi_flat[i] == 0.0
            s.flag[i] = FROZEN
            s.dist[i] = 0.0
        end
    end
    # Cells straddling the zero level set: interpolated crossing distance
    # per dimension, combined as sqrt(1/Σ 1/dᵢ²), signed by phi.
    for i in 1:n
        s.flag[i] == FAR || continue
        drow = 0.0
        dcol = 0.0
        borders = false
        for dir in (-1, 1)
            nrow = _neighbor(s, i, 1, dir, MASK)
            if nrow != 0 && s.phi_flat[i] * s.phi_flat[nrow] < 0
                borders = true
                d = s.dxrow * s.phi_flat[i] / (s.phi_flat[i] - s.phi_flat[nrow])
                if drow == 0 || drow > d
                    drow = d
                end
            end
            ncol = _neighbor(s, i, 2, dir, MASK)
            if ncol != 0 && s.phi_flat[i] * s.phi_flat[ncol] < 0
                borders = true
                d = s.dxcol * s.phi_flat[i] / (s.phi_flat[i] - s.phi_flat[ncol])
                if dcol == 0 || dcol > d
                    dcol = d
                end
            end
        end
        borders || continue
        dsum = 0.0
        drow > 0 && (dsum += 1 / drow / drow)
        dcol > 0 && (dsum += 1 / dcol / dcol)
        s.dist[i] = s.phi_flat[i] < 0 ? -sqrt(1 / dsum) : sqrt(1 / dsum)
        s.flag[i] = FROZEN
    end
    # travelTimeMarcher::initalizeFrozen
    for i in 1:n
        if s.flag[i] == FROZEN
            s.dist[i] = abs(s.dist[i] / s.speed_flat[i])
        end
    end
    return
end

# base_marcher.cpp initalizeNarrow: flat-order scan; each Far cell adjacent
# to a frozen cell is given its first (full-stencil) update and pushed.
function _initialize_narrow!(s::_Solver, h::_Heap)
    n = length(s.flag)
    for i in 1:n
        s.flag[i] == FAR || continue
        for dim in (1, 2), dir in (-1, 1)
            naddr = _neighbor(s, i, dim, dir, MASK)
            (naddr != 0 && s.flag[naddr] == FROZEN && s.flag[i] == FAR) || continue
            s.flag[i] = NARROW
            d = s.order == 2 ? _update_order_two_public(s, i) : _update_order_one(s, i)
            s.dist[i] = d
            s.heapptr[i] = _heap_push!(h, i, abs(d))
        end
    end
    return
end

# base_marcher.cpp solve(): pop-freeze (batch-freezing exact ties), then
# re-update the neighbors of every freshly frozen cell.
function _solve!(s::_Solver, h::_Heap)
    frozen_count = count(==(FROZEN), s.flag)
    frozen_count == 0 &&
        throw(FastMarchingError("the array phi contains no zero contour (no zero level set)"))
    while !_isempty(h)
        tofreeze = Int[]
        addr, value = _heap_pop!(h)
        s.flag[addr] = FROZEN
        push!(tofreeze, addr)
        # skfmm freezes every heap entry whose value is exactly equal in the
        # same batch, before any neighbor updates run.
        while !_isempty(h) && value == _peek(h)
            laddr, _ = _heap_pop!(h)
            s.flag[laddr] = FROZEN
            push!(tofreeze, laddr)
        end
        for a in tofreeze
            for dim in (1, 2), dir in (-1, 1)
                naddr = _neighbor(s, a, dim, dir, FROZEN)
                if naddr != 0 && s.flag[naddr] != FROZEN
                    if s.flag[naddr] == NARROW
                        d = s.order == 2 ? _update_order_two_public(s, naddr) :
                            _update_order_one(s, naddr)
                        if d != 0.0
                            _heap_set!(h, s.heapptr[naddr], abs(d))
                            s.dist[naddr] = d
                        end
                    elseif s.flag[naddr] == FAR
                        d = s.order == 2 ? _update_order_two_public(s, naddr) :
                            _update_order_one(s, naddr)
                        if d != 0.0
                            s.dist[naddr] = d
                            s.flag[naddr] = NARROW
                            s.heapptr[naddr] = _heap_push!(h, naddr, abs(d))
                        end
                    end
                end
                # Order-2 "jump": a narrow cell two away may now see the newly
                # frozen cell as its second-order far point.
                if s.order == 2
                    local_naddr = _neighbor(s, a, dim, dir, MASK)
                    if local_naddr != 0 && s.flag[local_naddr] == FROZEN
                        naddr2 = _neighbor(s, a, dim, 2 * dir, FROZEN)
                        if naddr2 != 0 && s.flag[naddr2] == NARROW
                            d = _update_order_two_public(s, naddr2)
                            if d != 0.0
                                _heap_set!(h, s.heapptr[naddr2], abs(d))
                                s.dist[naddr2] = d
                            end
                        end
                    end
                end
            end
        end
    end
    # base_marcher.cpp: anything never frozen is sentineled; skfmm's wrapper
    # masks these — the port returns NaN at the same positions.
    for i in eachindex(s.flag)
        if s.flag[i] != FROZEN
            s.dist[i] = MAX_DOUBLE
        end
    end
    return
end

function _travel_time(phi::Matrix{Float64}, speed::Matrix{Float64},
                      dxrow::Float64, dxcol::Float64, order::Int)
    nrows, ncols = size(phi)
    n = nrows * ncols
    # Flatten row-major (numpy C order) FIRST, then derive flags from the
    # flattened cells so matrix (column-major) and vector (row-major) index
    # spaces never mix.
    phi_flat = vec(permutedims(phi))
    speed_flat = vec(permutedims(speed))
    flag = fill(FAR, n)
    # pfmm.py: explicit masked-array masks become walls; the GeoNet pipeline
    # passes plain arrays, so only the zero-speed rule applies here
    # (travel_time_marcher.h constructor: speed < double-epsilon -> Mask).
    for i in 1:n
        speed_flat[i] < DOUBLE_EPSILON && (flag[i] = MASK)
    end
    s = _Solver(phi_flat, speed_flat,
                zeros(Float64, n), flag, zeros(Int, n),
                dxrow, dxcol, 1 / dxrow^2, 1 / dxcol^2, nrows, ncols, order)
    _initialize_frozen!(s)
    h = _Heap()
    _initialize_narrow!(s, h)
    _solve!(s, h)
    # Undo the row-major flattening; sentinel -> NaN where skfmm masks.
    result = Matrix{Float64}(undef, nrows, ncols)
    for row in 1:nrows, col in 1:ncols
        v = s.dist[(row - 1) * ncols + col]
        result[row, col] = v == MAX_DOUBLE ? NaN : v
    end
    return result
end

end # module
