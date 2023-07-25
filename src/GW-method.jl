"""
    sampling(a::Real)

sampling of z ~ g(z) ∝ 1/√a, a ∈ [a⁻¹, a]
"""
@inline sampling(a::Real)          = sampling(a, rand())
@inline sampling(a::Real, u::Real) = a * u * u - 2 * u * (u - 1) + abs2(u - 1) / a
