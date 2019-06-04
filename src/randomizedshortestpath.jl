abstract type Cost end
struct MinusLog <: Cost end
struct ExpMinus <: Cost end
struct Inv      <: Cost end

(::MinusLog)(x::Number) = -log(x)
(::ExpMinus)(x::Number) = exp(-x)
(::Inv)(x::Number)      = inv(x)

Base.inv(::MinusLog) = ExpMinus()
Base.inv(::ExpMinus) = MinusLog()
Base.inv(::Inv)      = Inv()

_Pref(A::SparseMatrixCSC) = Diagonal(inv.(vec(sum(A, dims=2)))) * A

function _W(Pref::SparseMatrixCSC, β::Real, C::SparseMatrixCSC)

    n = LinearAlgebra.checksquare(Pref)
    if LinearAlgebra.checksquare(C) != n
        throw(DimensionMismatch("Pref and C must have same size"))
    end

    return Pref .* exp.((-).(β) .* C)
end

function RSP_full_betweenness_qweighted(Z::AbstractMatrix,
                                        source_qualities::AbstractVector,
                                        target_qualities::AbstractVector)

    Zdiv = inv.(Z)

    ZQZdivQt = target_qualities .* Zdiv .* source_qualities'
    ZQZdivQt .-= Diagonal(sum(source_qualities) .* target_qualities .* diag(Zdiv))

    ZQZdivQt = Z*ZQZdivQt'

    return sum(ZQZdivQt' .* Z, dims=1)
end

function RSP_full_betweenness_kweighted(Z::AbstractMatrix,
                                        source_qualities::AbstractVector,
                                        target_qualities::AbstractVector,
                                        similarities::AbstractMatrix)

    Zdiv = inv.(Z)

    K = source_qualities .* similarities .* target_qualities'

    ZKZdiv = K .* Zdiv
    ZKZdiv -= Diagonal(vec(sum(K, dims=1)) .* diag(Zdiv))

    ZKZdiv = Z*ZKZdiv'

    return sum(ZKZdiv' .* Z, dims=1)
end

function RSP_dissimilarities(W::SparseMatrixCSC, C::SparseMatrixCSC, Z::AbstractMatrix = inv(Matrix(I - W)))
    S   = (Z*(C .* W)*Z) ./ Z
    d_s = diag(S)
    C̄   = S .- d_s'
    return C̄
end

RSP_free_energy_distance(Z::AbstractMatrix, β::Real) = -log.(Z*Diagonal(inv.(diag(Z))))/β
