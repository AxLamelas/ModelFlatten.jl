struct Fixed{T}
  val::T
end

const FixedLike = Union{Dirac,Fixed}

struct Indicator
  offset::Int
  length::Int
end

struct ArrayIndicator{D}
  offset::Int
  length::Int
  size::NTuple{D,Int}
end


struct Descriptor{NT,I}
  info::I
end

function Descriptor(nt::NamedTuple)
  info = create_info(nt)
  NT = create_type(nt)
  return Descriptor{NT,typeof(info)}(info)
end

function Base.length(desc::Descriptor)
  v = first(Iterators.filter( x -> !(x isa Fixed),Iterators.reverse(desc.info)))
  ind = v isa Tuple ? v[1] : v
  return ind.length + ind.offset 
end

(d::Descriptor)(x) = reconstruct(d,x)

function reconstruct(d::Descriptor{NT},x::AbstractVector{T}) where {NT,T}
  reconstruct(NT{T,1,Vector{T},Tuple{UnitRange{Int64}},true},d.info,x)
end

function reconstruct(d::Descriptor{NT},x::SubArray{T,N,P,I,L}) where {NT,T,N,P,I,L}
  reconstruct(NT{T,N,P,Tuple{UnitRange{Int64},Int64},L},d.info,x)
end

function reconstruct(::Type{W},info::TupleLike,x::AbstractVector{T}) where {W,T}
  W(if v isa Indicator
    if v.length == 1
      x[v.offset+1]
    else
      view(x,v.offset+1:v.offset+v.length)
    end
  elseif v isa ArrayIndicator
      reshape(view(x,v.offset+1:v.offset+v.length),v.size)
  elseif v isa Fixed
    v.val
  else
    reconstruct(
      W.parameters[2].parameters[i],
      v[2],
      view(x,v[1].offset+1:v[1].offset+v[1].length)
    )
  end for (i,v) in enumerate(values(info)))
end

create_info(nt::NamedTuple) = last(_create_info(nt))


function _create_info(nt::NamedTuple)
  offset = 0
  data = []
  for v in values(nt)
    if v isa TupleLike
      l, interior = _create_info(v)
      push!(data,(Indicator(offset,l),interior))
      offset += l
    elseif v isa Fixed
      push!(data,v)
    elseif v isa Dirac
      push!(data,Fixed(v.value))
    elseif v isa AbstractArray || v isa MultivariateDistribution
      push!(data,ArrayIndicator(offset,length(v),size(v)))
      offset += length(v)
    else
      push!(data,Indicator(offset,length(v)))
      offset += length(v)
    end
  end

  return offset, NamedTuple{keys(nt)}(data)
end

function _create_info(nt::Tuple)
  offset = 0
  data = []
  for v in values(nt)
    if v isa TupleLike
      l, interior = _create_info(v)
      push!(data,(Indicator(offset,l),interior))
      offset += l
    elseif v isa Fixed
      push!(data,v)
    elseif v isa Dirac
      push!(data,Fixed(v.value))
    elseif v isa AbstractArray || v isa MultivariateDistribution
      push!(data,ArrayIndicator(offset,length(v),size(v)))
      offset += length(v)
    else
      push!(data,Indicator(offset,length(v)))
      offset += length(v)
    end
  end

  return offset, Tuple(data)
end

function create_type(nt::NamedTuple)
  t = _create_type(nt)
  return Meta.eval(:($(t) where {W,N,P,I,L}))
end

function _create_type(nt::NamedTuple)
  t = :(Tuple{$((
    if v isa TupleLike
      _create_type(v)
    elseif v isa Fixed
      typeof(v.val)
    elseif v isa Dirac
      typeof(v.value)
    elseif v isa AbstractArray
      if isone(ndims(v))
        :(SubArray{W,N,P,I,L})
      else
        :(Base.ReshapedArray{W,$(ndims(v)),SubArray{W,N,P,I,L},Tuple{}})
      end
    elseif v isa MultivariateDistribution
        :(SubArray{W,N,P,I,L})
    else
    if length(v) == 1
      :W 
    else
        :(SubArray{W,N,P,I,L})
    end
    end
    for v in values(nt))...
  )})


  return :(NamedTuple{$(Tuple(keys(nt))),$(t)}) 
end

function _create_type(nt::Tuple)
  t = :(Tuple{$((
    if v isa TupleLike
      _create_type(v)
    elseif v isa Fixed
      typeof(v.val)
    elseif v isa Dirac
      typeof(v.value)
    elseif v isa AbstractArray
      if isone(ndims(v))
        :(SubArray{W,N,P,I,L})
      else
        :(Base.ReshapedArray{W,$(ndims(v)),SubArray{W,N,P,I,L},Tuple{}})
      end
    else
    if length(v) == 1
      :W 
    else
        :(SubArray{W,N,P,I,L})
    end
    end
    for v in nt)...
  )})


  return t
end

flat_eltype(desc::Descriptor,nt::TupleLike) = _flat_eltype(desc.info,nt)

function _flat_eltype(info::TupleLike,nt::TupleLike)
  types = ((begin
    v = getproperty(nt,k)
    if v isa TupleLike
      _flat_eltype(getproperty(info,k)[2],v)
    else
      eltype(v)
    end
  end for k in keys(info))...,)

  promote_type(types...)
end
