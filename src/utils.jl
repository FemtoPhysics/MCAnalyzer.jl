mean(x::VecI{T}) where T<:Real = sum(x) / length(x)

function dot(x::VecI{Tx}, y::VecI{Ty}, n::Int) where {Tx<:Real,Ty<:Real}
    r = 0.0
    m = mod(n, 5)
    if m ≠ 0
        @inbounds for i in 1:m
            r += x[i] * y[i]
        end
        n < 5 && return r
    end
    m += 1
    @inbounds for i in m:5:n
        r += x[i] * y[i] + x[i+1] * y[i+1] + x[i+2] * y[i+2] + x[i+3] * y[i+3] + x[i+4] * y[i+4]
    end
    return r
end

function dot(x::VecI{Tx}, y::VecI{Ty}) where {Tx<:Real,Ty<:Real}
    n = length(x)
    n ≡ length(y) || error("dot: length(y) ≠ length(x) = $n.")
    return dot(x, y, n)
end

"""
    welfordStep(μ::Real, s::Real, v::Real, c::Real)

Perform a single step of Welford algorithm (sample mean and variance)
- `μ` := sample mean
- `s` := sample variance
- `v` := c-th value
- `c` := c-th count
"""
function welfordStep(μ::Real, s::Real, v::Real, c::Real)
    isone(c) && return v, zero(v)
    s = s * (c - 2)
    m = μ + (v - μ) / c
    s = s + (v - μ) * (v - m)
    return m, s / (c - 1)
end

"""
    meanVarAdd(a::Real, μA::Real, σA²::Real, b::Real, μB::Real, σB²::Real)
"""
function meanVarAdd(a::Real, μA::Real, σA²::Real, b::Real, μB::Real, σB²::Real)
    return (a * μA + b * μB, a * a * σA² + b * b * σB²)
end

"""
    meanVarDiv(μA::Real, σA²::Real, μB::Real, σB²::Real)
"""
function meanVarDiv(μA::Real, σA²::Real, μB::Real, σB²::Real)
    avg = μA / μB
    var = abs2(avg) * (σA² / abs2(μA) + σB² / abs2(μB))
    return avg, var
end

"""
    sampling(iter::AbstractUnitRange{T}, exc::T)

sampling from `iter` except for `exc`
"""
function sampling(iter::AbstractUnitRange{T}, exc::T) where T # type-stability ✓
    ret = rand(iter)
    while ret ≡ exc
        ret = rand(iter)
    end
    return ret
end
