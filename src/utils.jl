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
