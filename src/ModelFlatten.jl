module ModelFlatten

export flatten, Descriptor, Fixed
export UnboundedBijector, TruncatedBijector, setup_transforms

using Reexport

@reexport using Bijectors
@reexport using Distributions

const TupleLike = Union{Tuple,NamedTuple}

include("descriptor.jl")
include("transforms.jl")

function flatten(nt::TupleLike)
  vcat((if v isa TupleLike
    flatten(v)
  else
    v
  end
    for v in values(nt) if !(v isa Fixed))...)
end

end # module ModelFlatten
