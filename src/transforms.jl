using Bijectors: Bijector, TruncatedBijector, Stacked, inverse


const UnboundedBijector{T} = TruncatedBijector{T,T}

function UnboundedBijector(::Type{T}) where {T <: Number}
    return Bijectors.TruncatedBijector(typemin(T),typemax(T))
end

Base.minimum(b::Bijectors.TruncatedBijector) = b.lb
Base.maximum(b::Bijectors.TruncatedBijector) = b.ub
Base.minimum(b::Bijectors.TruncatedBijector{<:AbstractFloat,<:Any}) = if isinf(b.lb)
    nextfloat(b.lb)
else
    b.lb
end
Base.maximum(b::Bijectors.TruncatedBijector{<:Any,<:AbstractFloat}) = if isinf(b.ub)
    prevfloat(b.ub)
else
    b.ub
end

setup_transforms(priors::NamedTuple) = setup_transforms(flatten(priors)[1]...)

function setup_transforms(priors...)
    bs = map(bijector,priors)
    ranges = UnitRange{Int64}[]
    idx = 1
    for p in priors
        len = length(p)
        push!(ranges, idx:idx+len-1)
        idx += len
    end

    sb = Stacked(Tuple(bs), Tuple(ranges))
    isb = inverse(sb)

    return sb,isb
end




