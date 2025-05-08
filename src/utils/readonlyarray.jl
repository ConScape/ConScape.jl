# This array is used to wrap TargetInit variables
# to prevent errors in equations by accidentally writing into
# objects shared accross multiple formulations.
struct ReadOnlyArray{T,N,A<:AbstractArray{T,N}} <: DenseArray{T,N}
    data::A
end
ReadOnlyArray(A::ReadOnlyArray) = A

Base.parent(x::ReadOnlyArray) = x.data
Base.size(x::ReadOnlyArray) = size(parent(x))
Base.getindex(x::ReadOnlyArray, i...) = getindex(parent(x), i...)
Base.unsafe_convert(::Type{Ptr{T}}, a::ReadOnlyArray{T}) where {T} = _unsafe_convert_error()
Base.unsafe_convert(::Type{Ptr{S}}, a::ReadOnlyArray{T}) where {S,T} = _unsafe_convert_error()

_unsafe_convert_error() =
    error("Cannot convert `ReadOnlyArray` to pointer. You may be trying to solve an `ldiv!` into a shared rhs varable. Copy the rhs array to a workspace before the solve.")