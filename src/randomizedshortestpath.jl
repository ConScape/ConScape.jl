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
                                        qˢ::AbstractVector,
                                        qᵗ::AbstractVector)

    Zⁱ = inv.(Z)

    qˢZⁱqᵗ = qˢ .* Zⁱ .* qᵗ'
    qˢZⁱqᵗ .-= Diagonal(sum(qˢ) .* qᵗ .* diag(Zⁱ))

    ZqˢZⁱqᵗ = Z*qˢZⁱqᵗ'

    return sum(ZqˢZⁱqᵗ' .* Z, dims=1) # diag(ZqˢZⁱqᵗ * Z)
end

function RSP_full_betweenness_kweighted(Z::AbstractMatrix,  # Fundamental matrix of non-absorbing paths
                                        qˢ::AbstractVector, # Tource qualities
                                        qᵗ::AbstractVector, # Target qualities
                                        S::AbstractMatrix)  # Matrix of similarities


    Zⁱ = inv.(Z)

    K = qˢ .* S .* qᵗ'

    KZⁱ = K .* Zⁱ
    KZⁱ -= Diagonal(vec(sum(K, dims=1)) .* diag(Zⁱ))

    ZKZⁱ = Z*KZⁱ'

    return sum(ZKZⁱ' .* Z, dims=1) # diag(KZⁱ * Z)
end

function RSP_dissimilarities(W::SparseMatrixCSC, C::SparseMatrixCSC, Z::AbstractMatrix = inv(Matrix(I - W)))
    S   = (Z*(C .* W)*Z) ./ Z
    d_s = diag(S)
    C̄   = S .- d_s'
    return C̄
end

RSP_free_energy_distance(Z::AbstractMatrix, β::Real) = -log.(Z*Diagonal(inv.(diag(Z))))/β

function RSP_functionality(source_qualities::AbstractVector,
                                        target_qualities::AbstractVector,
                                        similarities::AbstractMatrix)
    functionality = source_qualities .* similarities .* target_qualities'
    return vec(sum(functionality, dims=1))
end
