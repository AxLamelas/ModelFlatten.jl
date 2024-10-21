
struct Parameter{T<:Number,B<:Bijector}
  val::T
  bijector::B
end

function Parameter(val,lb,ub)
  v,l,u = promote(val,lb,ub)
  Parameter(v,TruncatedBijector(l,u))
end

Parameter(val,(lb,ub)) = Parameter(val,lb,ub)

function Parameter(val::T) where {T<:Number}
  Parameter(val,TruncatedBijector(typemin(T),typemax(T)))
end

value(p::Parameter) = p.val
Bijectors.bijector(p::Parameter) = p.bijector

Base.length(p::Parameter) = length(value(p))


function Base.show(io::IO, ::MIME"text/plain", p::Parameter)
  b = bijector(p)
  print(io,"P[$(value(p))] ($(minimum(b)), $(maximum(b)))")
end

function Base.show(io::IO, p::Parameter)
  print(io,"P[$(value(p))]")
end
