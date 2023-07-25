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
