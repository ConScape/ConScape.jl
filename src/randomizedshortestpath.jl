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
    Zdiv_diag = diag(Zdiv)

    qs_sum = sum(source_qualities)

    ZQZdivQ = target_qualities .* Zdiv'
    ZQZdivQ = ZQZdivQ .* source_qualities'
    ZQZdivQ -= Diagonal(qs_sum .* target_qualities .* Zdiv_diag)

    ZQZdivQ = Z*ZQZdivQ

    return sum(ZQZdivQ .* Z', dims=2)
end

function RSP_full_betweenness_kweighted(Z::AbstractMatrix,
                                        source_qualities::AbstractVector,
                                        target_qualities::AbstractVector,
                                        similarities::AbstractMatrix)

    Zdiv = inv.(Z)

    K = source_qualities .* similarities .* target_qualities'

    K_colsum = vec(sum(K, dims=1))
    d_Zdiv = diag(Zdiv)

    ZKZdiv = K .* Zdiv
    ZKZdiv -= Diagonal(K_colsum .* d_Zdiv)

    ZKZdiv = Z*ZKZdiv
    bet = sum(ZKZdiv .* Z', dims=1)

    return bet
end

function RSP_dissimilarities(W::SparseMatrixCSC, C::SparseMatrixCSC, Z::AbstractMatrix = inv(Matrix(I - W)))
    n   = LinearAlgebra.checksquare(W)
    CW  = C .* W
    S   = (Z*(C .* W)*Z) ./ Z
    d_s = diag(S)
    C̄   = S .- d_s
    return C̄
end

RSP_free_energy_distance(Z::AbstractMatrix, β::Real) = -log.(Z*Diagonal(inv.(diag(Z))))/β
