struct Fixed{T}
  val::T
end

struct Indicator
  offset::Int
  length::Int
end


struct Descriptor{NT,I}
  info::I
end

function Descriptor(nt::NamedTuple)
  info = create_info(nt)
  NT = create_type(nt)
  return Descriptor{NT,typeof(info)}(info)
end

function (d::Descriptor{NT})(x::AbstractVector{T}) where{NT,T}
  reconstruct(NT{T},d.info,x)
end

function reconstruct(d::Descriptor{NT},x::AbstractVector{T}) where {NT,T}
  reconstruct(NT{T},d.info,x)
end

function reconstruct(::Type{W},info::TupleLike,x::AbstractVector{T}) where {W,T}
  W(if v isa Indicator
    if v.length == 1
      x[v.offset+1]
    else
      view(x,v.offset+1:v.offset+v.length)
    end
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

create_info(nt::NamedTuple) = last(create_info(0,nt))


function create_info(offset::Int,nt::NamedTuple)
  data = []
  for v in values(nt)
    if v isa TupleLike
      l, interior = create_info(offset,v)
      push!(data,(Indicator(offset,l),interior))
    elseif v isa Fixed
      push!(data,v)
    else
      push!(data,Indicator(offset,length(v)))
      offset += length(v)
    end
  end

  return offset, NamedTuple{keys(nt)}(data)
end

function create_info(offset::Int,nt::Tuple)
  data = []
  for v in values(nt)
    if v isa TupleLike
      l, interior = create_info(offset,v)
      push!(data,(Indicator(offset,l),interior))
    elseif v isa Fixed
      push!(data,v)
    else
      push!(data,Indicator(offset,length(v)))
      offset += length(v)
    end
  end

  return offset, Tuple(data)
end

function create_type(nt::NamedTuple)
  t = _create_type(nt)
  return Meta.eval(:($(t) where {W}))
end

function _create_type(nt::NamedTuple)
  t = :(Tuple{$((
    if v isa TupleLike
      _create_type(v)
    elseif v isa Fixed
      typeof(v.val)
    else
    if length(v) == 1
      :W 
    else
        :(SubArray{W,1,Vector{W},Tuple{UnitRange{Int64}},true})
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
    else
    if length(v) == 1
      :W 
    else
        :(SubArray{W,1,Vector{W},Tuple{UnitRange{Int64}},true})
    end
    end
    for v in nt)...
  )})


  return t
end
