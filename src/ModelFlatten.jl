module ModelFlatten

export flatten, value, Parameter

using Reexport

@reexport using Bijectors
@reexport using Distributions

using Bijectors: Bijector, TruncatedBijector, Stacked, inverse

export UnboundedBijector, vector_and_transformation, TruncatedBijector, setup_transforms

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


abstract type AbstractParameter end

struct ArrayParameter{A<:AbstractArray{<:Number},B <: Bijector} <: AbstractParameter
    value::A
    bijector::B
end

struct Parameter{T,B} <: AbstractParameter
    value::T
    bijector::B
end

function Parameter(x,lb,ub)
    a,b,c = promote(x,lb,ub)
    Parameter(a,TruncatedBijector(b,c))
end

function ArrayParameter(x::AbstractArray{T},lb,ub) where {T}
    _,b,c = promote(zero(T),lb,ub)
    ArrayParameter(typeof(b).(x),TruncatedBijector(b,c))
end

Bijectors.bijector(x::AbstractParameter) = x.bijector
value(x::AbstractParameter) = x.value
Base.length(x::AbstractParameter) = length(value(x))
Base.show(io::IO, p::AbstractParameter) = print(io,"P[$(value(p))]")
Base.iterate(x::AbstractParameter) = (x,nothing)
Base.iterate(::AbstractParameter,::Nothing) = nothing
# Base.Broadcast.broadcastable(x::AbstractParameter) = (x,)

# Base.length(x::Dirichlet) = length(x.alpha)

const ParameterLike = Union{<:AbstractParameter,Distribution,Bijector}

@inline wrap(x) = if x isa ParameterLike || !applicable(iterate,x)
    (x,)
else
    x
end


@inline tuplejoin(x) = wrap(x)
@inline tuplejoin(x, y) = (wrap(x)...,wrap(y)...)
@inline tuplejoin(x, y, z...) = (wrap(x)..., tuplejoin(y, z...)...)

function _length(x)
    return if !applicable(iterate,x) 
        if !applicable(length,x)
            1
        else
            length(x)
        end
    elseif isempty(x)
        0
    elseif x isa Number
        return 1
    else
        sum(!applicable(length,xi) ? 1 : length(xi) for xi in x)
    end
end

function flatten(x,exclude_fixed::Bool)
    if exclude_fixed
        unflatten_Fixed(_) = x
        return (), unflatten_Fixed
    else
        unflatten(v) = only(v)
        return x, unflatten
    end
end

function flatten(x::ParameterLike,exclude_fixed::Bool)
    unflatten(v) = only(v)
    return x, unflatten
end

function flatten(x::Distribution{Univariate},exclude_fixed::Bool)
    unflatten_univariate(v) = only(v)
    return x, unflatten_univariate
end

function flatten(x::Distribution{Multivariate},exclude_fixed::Bool)
    unflatten_multvariate(v) = v[1:length(x)]
    return x, unflatten_multvariate
end

function flatten(x::AbstractArray,exclude_fixed::Bool) 
    x_vecs_and_backs = map(val -> flatten(val,exclude_fixed), x)
    x_vecs, x_backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    lengths = reshape(map(_length,x_vecs),:)
    sz = cumsum(lengths)
    function unflatten_to_vec(v)
        reshape(map(x_backs, lengths, sz) do x_back, l, s
            return x_back(v[(s - l + 1):s])
        end,size(x)...)
    end
    return tuplejoin(x_vecs...), unflatten_to_vec
end


function flatten(x::AbstractVector,exclude_fixed::Bool) 
    x_vecs_and_backs = map(val -> flatten(val,exclude_fixed), x)
    x_vecs, x_backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    lengths = map(_length,x_vecs)
    sz = cumsum(lengths)
    function unflatten_to_vec(v)
        map(x_backs, lengths, sz) do x_back, l, s
            return x_back(v[(s - l + 1):s])
        end
    end
    return tuplejoin(x_vecs...), unflatten_to_vec
end

function flatten(x::Tuple,exclude_fixed::Bool) 
    x_vecs_and_backs = map(val -> flatten(val,exclude_fixed), x)
    x_vecs, x_backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    lengths = map(_length,x_vecs)
    sz = cumsum(lengths)
    function unflatten_to_Tuple(v)
        map(x_backs, lengths, sz) do x_back, l, s
            return x_back(v[(s - l + 1):s])
        end
    end
    return tuplejoin(x_vecs...), unflatten_to_Tuple
end

function flatten(x::NamedTuple{N},exclude_fixed::Bool) where {N}
    x_vec, unflatten = flatten(values(x),exclude_fixed)
    function unflatten_to_NamedTuple(v)
        v_vec_vec = unflatten(v)
        return NamedTuple{N}(v_vec_vec)
    end
    return x_vec, unflatten_to_NamedTuple
end

flatten(x) = flatten(x,true)


function vector_and_transformation(θ0::NTuple{N,T}) where {N,T <: Union{Parameter,ArrayParameter}}
    bijectors = bijector.(θ0)

    to_unconstrained = Stacked(bijectors)
    to_constrained = inverse(to_unconstrained)

    return (;θ0 = to_unconstrained(vcat(value.(θ0)...)), to_unconstrained, to_constrained)
end

function vector(θ0::NTuple{N,T}) where {N,T <: Union{Parameter,ArrayParameter}}
    bijectors = bijector.(θ0)
    to_unconstrained = Stacked(bijectors)

    return to_unconstrained(vcat(value.(θ0)...))
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



end # module ModelFlatten
