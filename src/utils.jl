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

autodot(x::VecI{Tx}, T::Int, τ::Int) where Tx<:Real = dot(view(x, 1:(T-τ)), view(x, (τ+1):T))

"""
    autocor!(cor::VecIO{Tr}, sig::VecI{Tx}, dev::VecB{Tx}) where {Tr<:Real, Tx<:Real}

Compute a real-valued signal vector's autocorrelation function (ACF) in a nonnegative lag regime.
The mean of signals would be subtracted from the signal vector before computing the ACF and stored in the provided buffer vector.
The output is normalized by the variance of signals, i.e., the zero-lag autocorrelation is 1.
"""
function autocor!(cor::VecIO{Tr}, sig::VecI{Tx}, dev::VecB{Tx}) where {Tr<:Real, Tx<:Real}
    size_sig = length(sig)
    size_sig ≡ length(dev) || throw(DimensionMismatch())
    size_sig ≡ length(cor) || throw(DimensionMismatch())

    avg = mean(sig)
    @simd for i in eachindex(dev)
        @inbounds dev[i] = sig[i] - avg
    end

    var = dot(dev, dev)
    @inbounds for i in eachindex(cor)
        cor[i] = autodot(dev, size_sig, i - 1) / var
    end
    return cor
end
