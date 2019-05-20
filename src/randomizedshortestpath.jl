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

function RSP_full_betweenness_qweighted(Z::AbstractMatrix,
                                        source_qualities::AbstractMatrix,
                                        target_qualities::AbstractMatrix)

    Zdiv = inv.(Z)
    Zdiv_diag = diag(Zdiv)

    qs = vec(source_qualities)
    qt = vec(target_qualities)
    qs_sum = sum(qs)

    ZQZdivQ = qt .* Zdiv'
    ZQZdivQ = ZQZdivQ .* qs'
    ZQZdivQ -= Diagonal(qs_sum .* qt .* Zdiv_diag)

    ZQZdivQ = Z*ZQZdivQ

    return reshape(sum(ZQZdivQ .* Z', dims=2), size(source_qualities)...)
end

function RSP_full_betweenness_kweighted(Z::AbstractMatrix,
                                        source_qualities::AbstractMatrix,
                                        target_qualities::AbstractMatrix,
                                        similarities::AbstractMatrix)

    Zdiv = inv.(Z)

    qs = vec(source_qualities)
    qt = vec(target_qualities)

    K = qs .* similarities .* qt'

    K_colsum = vec(sum(K, dims=1))
    d_Zdiv = diag(Zdiv)

    ZKZdiv = K .* Zdiv
    ZKZdiv -= Diagonal(K_colsum .* d_Zdiv)

    ZKZdiv = Z*ZKZdiv
    bet = sum(ZKZdiv .* Z', dims=1)

    return reshape(bet, size(source_qualities))
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
