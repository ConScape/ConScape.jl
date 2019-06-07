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

    ZqˢZⁱqᵗZ = Zⁱ # reuse Zⁱ memory
    mul!(ZqˢZⁱqᵗZ, Z, qˢZⁱqᵗ')
    ZqˢZⁱqᵗZ .*= Z'

    return sum(ZqˢZⁱqᵗZ, dims=2) # diag(ZqˢZⁱqᵗ * Z)
end

function RSP_full_betweenness_kweighted(Z::AbstractMatrix,  # Fundamental matrix of non-absorbing paths
                                        qˢ::AbstractVector, # Source qualities
                                        qᵗ::AbstractVector, # Target qualities
                                        S::AbstractMatrix)  # Matrix of similarities


    Zⁱ = inv.(Z)

    KZⁱ = qˢ .* S .* qᵗ'
    k = vec(sum(KZⁱ, dims=1))

    KZⁱ .*= Zⁱ
    for i in 1:size(KZⁱ, 1)
        KZⁱ[i, i] -= k[i] .* Zⁱ[i, i]
    end

    ZKZⁱ = Zⁱ # reuse Zⁱ memory
    mul!(ZKZⁱ, Z, KZⁱ')
    ZKZⁱ .*= Z'

    return sum(ZKZⁱ, dims=2) # diag(KZⁱ * Z)
end

function RSP_dissimilarities(W::SparseMatrixCSC, C::SparseMatrixCSC, Z::AbstractMatrix = inv(Matrix(I - W)))
    C̄   = Z*(C .* W)*Z
    C̄ ./= Z
    dˢ  = diag(C̄)
    C̄ .-= dˢ'
    return C̄
end

RSP_free_energy_distance(Z::AbstractMatrix, β::Real) = -log.(Z*Diagonal(inv.(diag(Z))))/β

function RSP_functionality(qˢ::AbstractVector, # Source qualities
                           qᵗ::AbstractVector, # Target qualities
                           S::AbstractMatrix)  # Matrix of similarities

    return (S'*qˢ) .* qᵗ
end
