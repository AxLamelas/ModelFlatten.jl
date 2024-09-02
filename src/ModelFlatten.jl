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

const ArrayParameter{T,B} = Tuple{A,B} where {T <: Number, A <: AbstractArray{T}, B <: Bijector}
const Parameter{T,B} = Tuple{T,B} where {T <: Number, B <: Bijector}
Base.length(x::Parameter) = length(x[1])
Base.show(io::IO, p::Parameter) = print(io,"P[$(p[1])]")
Bijectors.bijector(x::Parameter) = x[2]
value(x::Parameter) = x[1]

# Base.length(x::Dirichlet) = length(x.alpha)

const ParameterLike = Union{ArrayParameter, Parameter,Distribution,Bijector}

@inline wrap(x) = if x isa ParameterLike
    (x,)
else
    x
end
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (wrap(x)...,wrap(y)...)
@inline tuplejoin(x, y, z...) = (wrap(x)..., tuplejoin(y, z...)...)

function _length(x)
    return if !applicable(iterate,x) 
        length(x)
    elseif isempty(x)
        0
    else
        sum(length(xi) for xi in x)
    end
end

function flatten(x)
    unflatten_Fixed(_) = x
    return () , unflatten_Fixed
end

function flatten(x::ParameterLike)
    unflatten(v) = only(v)
    return x, unflatten
end

function flatten(x::Distribution{Univariate})
    unflatten_univariate(v) = only(v)
    return x, unflatten_univariate
end

function flatten(x::Distribution{Multivariate})
    unflatten_multvariate(v) = v[1:length(x)]
    return x, unflatten_multvariate
end

function flatten(x::AbstractArray) 
    x_vecs_and_backs = map(val -> flatten(val), x)
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


function flatten(x::AbstractVector) 
    x_vecs_and_backs = map(val -> flatten(val), x)
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

function flatten(x::Tuple) 
    x_vecs_and_backs = map(val -> flatten(val), x)
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

function flatten(x::NamedTuple{N}) where {N}
    x_vec, unflatten = flatten(values(x))
    function unflatten_to_NamedTuple(v)
        v_vec_vec = unflatten(v)
        return NamedTuple{N}(v_vec_vec)
    end
    return x_vec, unflatten_to_NamedTuple
end

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

function setup_transforms(priors::NamedTuple)
    bs = bijector.(values(priors))
    ranges = UnitRange{Int64}[]
    idx = 1
    for p in priors
        len = length(p)
        push!(ranges, idx:idx+len-1)
        idx += len
    end

    sb = Stacked(bs, ranges)
    isb = inverse(sb)

    return sb, isb
end



end # module ModelFlatten
