using Bijectors: Bijector, TruncatedBijector, Stacked, inverse

using Base: AbstractVecOrTuple

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

function Base.minimum(b::Bijectors.Stacked)
    vcat((begin
        m = minimum(b)
        if length(m) == length(r)
            m
        else
            fill(m,length(r))
        end
    end for (r,b) in zip(b.ranges_in,b.bs))...)
end

function Base.maximum(b::Bijectors.Stacked)
    vcat((begin
        m = maximum(b)
        if length(m) == length(r)
            m
        else
            fill(m,length(r))
        end
    end for (r,b) in zip(b.ranges_in,b.bs))...)
end


function Bijectors.bijector(d::ContinuousMultivariateDistribution)
    m = minimum(d)
    M = maximum(d)

    return Stacked([TruncatedBijector(mi,Mi) for (mi,Mi) in zip(m,M)])
end

setup_transforms(nt::NamedTuple) = setup_transforms(flatten(nt)...)

setup_transforms(d::Distributions.ProductNamedTupleDistribution) = setup_transforms(values(d.dists)...)
setup_transforms(ps...) = setup_transforms(map(length,ps),map(bijector,ps))

function setup_transforms(lengths::AbstractVecOrTuple{Int},bs)
    ranges = UnitRange{Int64}[]
    idx = 1
    for len in lengths
        push!(ranges, idx:idx+len-1)
        idx += len
    end

    sb = Stacked(Tuple(bs), Tuple(ranges))
    isb = inverse(sb)

    return sb,isb
end




