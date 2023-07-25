"""
    sampling(a::Real)

sampling of z ~ g(z) ∝ 1/√a, a ∈ [a⁻¹, a]
"""
@inline sampling(a::Real)          = sampling(a, rand())
@inline sampling(a::Real, u::Real) = a * u * u - 2 * u * (u - 1) + abs2(u - 1) / a

"""
    stretchMove!(walker_new::VecI, walker_old::VecI, walker_com::VecI, a::Real)

- `walker_new` := the k-th walker's new vector
- `walker_old` := the k-th walker's old vector
- `walker_com` := the randomly chosen walker's vector from complementary ensemble of the k-th walker
"""
function stretchMove!(walker_new::VecI, walker_old::VecI, walker_com::VecI, a::Real)
    z = sampling(a)
    @simd for i in eachindex(walker_new)
        @inbounds walker_new[i] = walker_com[i] + z * (walker_old[i] - walker_com[i])
    end
    return z
end

"""
    logAcceptance(logpr_new::Real, logpr_old::Real, nx::Int, z::Real)
"""
@inline logAcceptance(logpr_new::Real, logpr_old::Real, nx::Int, z::Real) =
    min(0.0, (nx - 1) * log(z) + logpr_new - logpr_old)

"""
    MCMCSampler{Tc,Tp}(nx::Int, nw::Int, nt::Int) where {Tc<:Real,Tp<:Real}

To construct

    struct MCMCSampler{Tc,Tp}
        chain::Array{Tc,3}
        ρxsum::Array{Tc,3}
        logpr::Array{Tp,2}
    end

- `nx` := dim. of hidden states `{Xₜ}`
- `nw` := num. of walkers in the ensemble
- `nt` := num. of observation epochs
"""
struct MCMCSampler{Tc,Tp}
    chain::Array{Tc,3}
    ρxsum::Array{Tc,3}
    logpr::Array{Tp,2}

    MCMCSampler{Tc,Tp}(nx::Int, nw::Int, nt::Int) where {Tc<:Real,Tp<:Real} =
        new{Tc,Tp}(
            Array{Tc,3}(undef, nx, nw, nt),
            Array{Tc,3}(undef, nt,  4, nx),
            Array{Tp,2}(undef, nt, nw)
        )
end

"""
    initialize!(s::MCMCSampler, avg::NTuple{N,A}, std::NTuple{N,S}, log_pdf::Function) where {N,A<:Real,S<:Real}
"""
function initialize!(s::MCMCSampler, avg::NTuple{N,A}, std::NTuple{N,S}, log_pdf::Function) where {N,A<:Real,S<:Real}
    chain, logpr = s.chain, s.logpr
    initNormal!(view(chain, :, :, 1), avg, std)
    @inbounds for k in axes(logpr, 2)
        logpr[1,k] = log_pdf(view(chain, :, k, 1))
    end
    return nothing
end

"""
    initNormal!(des::MatI, avg::NTuple{N,A}, std::NTuple{N,S}) where {N,A<:Real,S<:Real}
"""
function initNormal!(des::MatI, avg::NTuple{N,A}, std::NTuple{N,S}) where {N,A<:Real,S<:Real}
    N ≡ size(des, 1) || error("normal_init!: N ≠ size(des, 1)")
    for k in axes(des, 2)
        @simd for i in axes(des, 1)
            @inbounds des[i,k] = avg[i] + randn() * std[i]
        end
    end
    return nothing
end

"""
    updateMCMC!(walkers_new::MatI, logprs_new::VecI,
                walkers_old::MatI, logprs_old::VecI,
                log_pdf::Function, nx::Int,
                one2K::Base.OneTo{Int}, a::Real)
"""
function updateMCMC!(walkers_new::MatI, logprs_new::VecI,
                     walkers_old::MatI, logprs_old::VecI,
                     log_pdf::Function, nx::Int,
                     one2K::Base.OneTo{Int}, a::Real)
    for k in one2K
        walker_new = view(walkers_new, :, k)
        walker_old = view(walkers_old, :, k)

        z = stretchMove!(
            walker_new, walker_old,
            view(walkers_old, :, sampling(one2K, k)), a
        )

        logpr_new = log_pdf(walker_new)
        logpr_old = @inbounds logprs_old[k]
        q = logAcceptance(logpr_new, logpr_old, nx, z)

        if log(rand()) > q
            copyto!(walker_new, walker_old)
            @inbounds logprs_new[k] = logpr_old
        else
            @inbounds logprs_new[k] = logpr_new
        end
    end
    return nothing
end

"""
    integrateAutocor!(sampler::MCMCSampler)
"""
function integrateAutocor!(sampler::MCMCSampler)
    chain = sampler.chain
    ρxsum = sampler.ρxsum

    one2N = axes(chain, 1)
    one2T = axes(ρxsum, 1)
    K = size(chain, 2)
    # Compute autocor. for each walker use view(ρsum, n, 3, :) as buffer
    # Compute avg. and var. over each walker for each dim. params.
    for n in one2N, k in 1:K
        autocor!(view(ρxsum, :, 3, n), view(chain, n, k, :), view(ρxsum, :, 4, n))
        @inbounds for t in one2T
            ρxsum[t,1,n], ρxsum[t,2,n] = welfordStep(
                ρxsum[t,1,n], ρxsum[t,2,n], ρxsum[t,3,n], k
            )
        end
    end

    # Compute Integrated autocor.
    @inbounds for n in one2N, t in 2:size(ρxsum, 1)
        ρxsum[t,1,n], ρxsum[t,2,n] = meanVarAdd(
            1.0, ρxsum[t-1,1,n], ρxsum[t-1,2,n],
            2.0, ρxsum[t,1,n],   ρxsum[t,2,n]
        )
    end

    @inbounds for n in one2N, t in one2T
        ρxsum[t,1,n], ρxsum[t,2,n] = meanVarDiv(ρxsum[t,1,n], ρxsum[t,2,n], t, 0.0)
    end

    factor = 2.5 / sqrt(K)
    @inbounds for n in one2N, t in one2T
        temp = factor * sqrt(ρxsum[t,2,n])
        ρxsum[t,3,n] = ρxsum[t,1,n] - temp
        ρxsum[t,4,n] = ρxsum[t,1,n] + temp
    end

    return nothing
end

"""
    estimateBias(sampler::MCMCSampler; C::Int=5)

Estimate the relaxation time of the discrete autocorrelation time-delay series according to the relation

    M ≥ C ⋅ τf(M)

, where `τf(M)` is the cumulative summation of the autocorrelation time-delay series up to `M` delay.
"""
function estimateBias(sampler::MCMCSampler; C::Int=5)
    ρsum = sampler.ρxsum
    invC = inv(C)

    ret = 0
    for n in axes(ρsum, 3)
        ret = max(ret, estimateBias(view(ρsum, :, 1, n), invC))
    end
    return ret
end

function estimateBias(ρsum::VecI, invC::Real)
    for i in eachindex(ρsum)
        if @inbounds ρsum[i] < invC
            return i
        end
    end
    return length(ρsum)
end

"""
    collect!(des::MatIO, sampler::MCMCSampler, bias::Int)
"""
function collect!(des::MatIO, sampler::MCMCSampler, bias::Int)
    chain = sampler.chain
    collect!(des, view(chain, :, :, bias:size(chain, 3)))
    return nothing
end

function collect!(des::MatIO, src::ArrIO{T,3}) where T<:Real
    cnt = 0
    for k in axes(src, 2), t in axes(src, 3)
        cnt += 1
        @inbounds for n in axes(src, 1)
            des[n,1], des[n,2] = welfordStep(
                des[n,1], des[n,2], src[n,k,t], cnt
            )
        end
    end

    fac = 2.5 / sqrt(cnt)
    @inbounds for n in axes(des, 1)
        tmp = fac * sqrt(des[n,2])
        des[n,3] = des[n,1] - tmp
        des[n,4] = des[n,1] + tmp
    end

    return nothing
end
