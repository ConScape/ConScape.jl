using LinearAlgebra

import Base: -

const Stencil2D9P{T} = SymTridiagonal{SymTridiagonal{T,Vector{T}},Vector{SymTridiagonal{T,Vector{T}}}}
const Stencil2D9P2{T} = SymTridiagonal{Tridiagonal{T,Vector{T}},Vector{Tridiagonal{T,Vector{T}}}}

function Base.show(io::IO, ::MIME"text/plain", S::Stencil2D9P)
    n = size(S.dv[1], 1)*length(S.dv)
    println(io, "Stencil2D9P of size ($n, $n)")
end

function Base.show(io::IO, ::MIME"text/plain", S::Stencil2D9P2)
    n = size(S.dv[1], 1)*length(S.dv)
    println(io, "Stencil2D9P2 of size ($n, $n)")
end

# From Meurant 1992 p. 714-716
function Base.inv(S::Stencil2D9P{T}) where T
    Ds = S.dv
    As = S.ev
    n  = length(Ds)

    Δs    = Array{Matrix{T}}(undef, n)
    Δs[1] = Ds[1]

    for i in 2:n
        Δs[i] = Ds[i] - As[i-1]*(Δs[i-1]\Matrix(As[i-1]'))
    end

    Σs    = Array{Matrix{T}}(undef, n)
    Σs[n] = Ds[n]
    for i in (n-1):-1:1
        Σs[i] = Ds[i] - As[i]'*(Σs[i+1]\Matrix(As[i]))
    end

    Us    = Array{Matrix{T}}(undef, n)
    Us[1] = Matrix{T}(I, size(Ds[1])...)

    for i in 2:n
        Us[i] = Us[i-1]*(-As[i-1]'\Δs[i-1])
    end

    Vs    = Array{Matrix{T}}(undef, n)
    Vs[1] = inv(Σs[1])

    for i in 2:n
        Vs[i] = Vs[i-1]*(Matrix(-As[i-1])'/Σs[i])
    end

    return Us, Vs, Δs, Σs
end


function Base.inv(S::Stencil2D9P2{T}) where T
    Ds = S.dv
    As = S.ev
    n  = length(Ds)

    Δs    = Array{Matrix{T}}(undef, n)
    Δs[1] = Ds[1]

    for i in 2:n
        Δs[i] = Ds[i] - As[i-1]*(Δs[i-1]\Matrix(As[i-1]'))
    end

    Σs    = Array{Matrix{T}}(undef, n)
    Σs[n] = Ds[n]
    for i in (n-1):-1:1
        Σs[i] = Ds[i] - As[i]'*(Σs[i+1]\Matrix(As[i]))
    end

    Us    = Array{Matrix{T}}(undef, n)
    Us[1] = Matrix{T}(I, size(Ds[1])...)

    for i in 2:n
        Us[i] = Us[i-1]*(-As[i-1]'\Δs[i-1])
    end

    Vs    = Array{Matrix{T}}(undef, n)
    Vs[1] = inv(Σs[1])

    for i in 2:n
        Vs[i] = Vs[i-1]*(Matrix(-As[i-1])'/Σs[i])
    end

    return Us, Vs, Δs, Σs
end

# # FIXME! For now we just wrap a SparseMatrixCSC but we should eventually come up with something better that enforces the pentadiagonal structure
# struct FivePointStencilMatrix{T} <: AbstractMatrix{T}
#     data::SparseMatrixCSC{T,Int}
#     rows::Int
#     cols::Int
#     function FivePointStencilMatrix(data::SparseMatrixCSC{T}, rows::Integer, cols::Integer) where T
#         n = rows*cols
#         _n = LinearAlgebra.checksquare(data)
#         if n != _n
#             throw(DimensionMismatch("size of sparse matrix must match producs of grid dimensions"))
#         end
#         # FIXME! Come up with a better check
#         n2 = div(n, 2)
#         for i in n2:(n2 + 10)
#             ndiags = count(!iszero, view(data, :, i))
#             if ndiags > 5
#                 throw(ArgumentError("matrix must be pentadiagonal but it has at least $ndiags diagonals"))
#             end
#         end
#         return new{T}(data, rows, cols)
#     end
# end
#
# Base.size(S::FivePointStencilMatrix) = size(S.data)
# Base.getindex(S::FivePointStencilMatrix, i::Integer...) = getindex(S.data, i...)
#
# ## structured matrix methods ##
# function Base.replace_in_print_matrix(S::FivePointStencilMatrix,
#                                       i::Integer, j::Integer, s::AbstractString)
#     if abs(i - j) <= 1 || abs(i - j) == S.rows
#         return s
#     else
#         return Base.replace_with_centered_mark(s)
#     end
# end
#
# (-)(S::FivePointStencilMatrix) = FivePointStencilMatrix(-S.data, S.rows, S.cols)
# (-)(S::FivePointStencilMatrix, J::UniformScaling) = FivePointStencilMatrix(S.data - J, S.rows, S.cols)
# (-)(J::UniformScaling, S::FivePointStencilMatrix) = FivePointStencilMatrix(J - S.data, S.rows, S.cols)
#
# struct InverseFivePointStencil{T} <: AbstractMatrix{T}
#     U::Matrix{T}
#     V::Matrix{T}
# end
#
# Base.size(IF::InverseFivePointStencil) = (size(IF.U, 1), size(IF.U, 1))
#
# function Base.getindex(IF::InverseFivePointStencil, i::Integer, j::Integer)
#     # if i < j
#     return dot(view(IF.U, i, :), view(IF.V, j, :))
# end
#
# function Base.inv(F::FivePointStencilMatrix)
#     n = size(F, 1)
#     issymmetric(F.data) || throw(ArgumentError("Currently only symmetric matrices are supported"))
#     F = lu(F.data)
#     V = F\[I; zeros(n - 3, 3)]
#     tmp = F\[zeros(n - 3, 3); I]
#     U = tmp/tmp[1:3,:];
#     return InverseFivePointStencil(U, V)
# end
