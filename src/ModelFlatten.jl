module ModelFlatten

export flatten, Descriptor, Fixed
export UnboundedBijector, TruncatedBijector, setup_transforms
export Parameter, value

using Reexport

@reexport using Bijectors
@reexport using Distributions

const TupleLike = Union{Tuple,NamedTuple}

include("descriptor.jl")
include("transforms.jl")
include("parameter.jl")

function flatten(nt::TupleLike)
  vcat((if v isa TupleLike
    flatten(v)
  else
    v
  end
    for v in values(nt) if !(v isa FixedLike))...)
end

function flatten(desc::Descriptor,nt::TupleLike) 
  T = flat_eltype(desc,nt)
  l = length(desc)
  r = Vector{T}(undef,l)
  _flatten!(r,desc.info,nt)
  return r
end

flatten!(r::AbstractVector,desc::Descriptor,nt::TupleLike) = _flatten!(r,desc.info,nt)

function _flatten!(r::AbstractVector,info::TupleLike,nt::TupleLike)
  for (k,info_v) in pairs(info)
    # If it is a FixedLike nothing is done
    if info_v isa Fixed continue end
    v = getproperty(nt,k)
    if info_v isa TupleLike
      ind,info_subset = info_v
      _flatten!(view(r,1+ind.offset:ind.offset+ind.length),info_subset,v)
    elseif info_v isa Union{Indicator,ArrayIndicator}
      ind = info_v
      r[1+ind.offset:ind.offset+ind.length] .= v

    end
  end
end



end # module ModelFlatten
