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
