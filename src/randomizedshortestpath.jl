_Pref(A::SparseMatrixCSC) = Diagonal(inv.(vec(sum(A, dims=2)))) * A

function _W(Pref::SparseMatrixCSC, θ::Real, C::SparseMatrixCSC)

    n = LinearAlgebra.checksquare(Pref)
    if LinearAlgebra.checksquare(C) != n
        throw(DimensionMismatch("Pref and C must have same size"))
    end

    W = Pref .* exp.((-).(θ) .* C)
    replace!(W.nzval, NaN => 0.0)

    return W
end

function RSP_betweenness_qweighted(W::SparseMatrixCSC,
                                   Z::AbstractMatrix,
                                   qˢ::AbstractVector,
                                   qᵗ::AbstractVector,
                                   targetnodes::AbstractVector)

    Zⁱ = inv.(Z)
    Zⁱ[.!isfinite.(Zⁱ)] .= floatmax(eltype(Z)) # To prevent Inf*0 later...

    qˢZⁱqᵗ = qˢ .* Zⁱ .* qᵗ'
    sumqˢ = sum(qˢ)
    for j in axes(Z, 2)
        qˢZⁱqᵗ[targetnodes[j], j] -=  sumqˢ * qᵗ[j] * Zⁱ[targetnodes[j], j]
    end

    ZqˢZⁱqᵗZt = (I - W)'\qˢZⁱqᵗ
    ZqˢZⁱqᵗZt .*= Z

    return sum(ZqˢZⁱqᵗZt, dims=2) # diag(Z * ZqˢZⁱqᵗ')
end


function RSP_betweenness_kweighted(W::SparseMatrixCSC,
                                   Z::AbstractMatrix,  # Fundamental matrix of non-absorbing paths
                                   qˢ::AbstractVector, # Source qualities
                                   qᵗ::AbstractVector, # Target qualities
                                   S::AbstractMatrix,  # Matrix of proximities
                                   landmarks::AbstractVector)


    axis1, axis2 = axes(Z)
    if axis1 != axes(qˢ, 1)
        throw(DimensionMismatch(""))
    end
    if axis2 != axes(qᵗ, 1)
        throw(DimensionMismatch(""))
    end
    if axes(S) != (axis1, axis2)
        throw(DimensionMismatch(""))
    end
    if axis2 != axes(landmarks, 1)
        throw(DimensionMismatch(""))
    end

    Zⁱ = inv.(Z)
    Zⁱ[.!isfinite.(Zⁱ)] .= floatmax(eltype(Z)) # To prevent Inf*0 later...

    KZⁱ = qˢ .* S .* qᵗ'

    # If any of the values of KZⁱ is above one then there is a risk of overflow.
    # Hence, we scale the matrix and apply the scale factor by the end of the
    # computation.
    λ = max(1.0, maximum(KZⁱ))
    k = vec(sum(KZⁱ, dims=1)) * inv(λ)

    KZⁱ .*= inv.(λ) .* Zⁱ
    for j in axis2
        KZⁱ[landmarks[j], j] -= k[j] .* Zⁱ[landmarks[j], j]
    end

    ZKZⁱt = (I - W)'\KZⁱ
    ZKZⁱt .*= λ .* Z

    return vec(sum(ZKZⁱt, dims=2)) # diag(Z * KZⁱ')
end

function RSP_edge_betweenness_qweighted(W::SparseMatrixCSC,
                                        Z::AbstractMatrix,
                                        qˢ::AbstractVector,
                                        qᵗ::AbstractVector,
                                        targetnodes::AbstractVector)

    Zⁱ = inv.(Z)
    Zⁱ[.!isfinite.(Zⁱ)] .= floatmax(eltype(Z)) # To prevent Inf*0 later...

    # FIXME: This should be only done when actually size(Z,2) < size(Z,1)/K where K ≈ 10 or so.
    # Otherwise we just compute many of the elements of Z twice...
    if size(Z,2) < size(Z,1)
        Zrows = ((I - W')\Matrix(sparse(targetnodes,
                                       1:length(targetnodes),
                                       1.0,
                                       size(W, 1),
                                       length(targetnodes))))'
    else
        Zrows = Z
    end

    n = size(W,1)


    diagZⁱ = [Zⁱ[targetnodes[t], t] for t in 1:length(targetnodes)]
    sumqˢ = sum(qˢ)

    Zrows = Zrows .* (sumqˢ*qᵗ.*diagZⁱ)

    qˢZⁱqᵗ = qˢ .* Zⁱ .* qᵗ'

    QZⁱᵀZ = qˢZⁱqᵗ'/(I - W)

    RHS = QZⁱᵀZ-Zrows

    edge_betweennesses = copy(W)

    for i in axes(W, 1)
        # ZᵀZⁱ_minus_diag = Z[:,i]'*qˢZⁱqᵗ .- sumqˢ.* (Z[:,i].*diag(Zⁱ).*qᵗ)'

        for j in findall(W[i,:].>0)
            # edge_betweennesses[i,j] = W[i,j] .* Zqt[j,:]'* (ZᵀZⁱ_minus_diag * Z[j,:])[1]
            edge_betweennesses[i,j] = W[i,j] .* (Z[j,:]' * RHS[:,i])[1]
        end
    end

    return edge_betweennesses
end

function RSP_edge_betweenness_kweighted(W::SparseMatrixCSC,
                                        Z::AbstractMatrix,
                                        qˢ::AbstractVector,
                                        qᵗ::AbstractVector,
                                        K::AbstractMatrix,  # Matrix of proximities
                                        targetnodes::AbstractVector)

    Zⁱ = inv.(Z)
    Zⁱ[.!isfinite.(Zⁱ)] .= floatmax(eltype(Z)) # To prevent Inf*0 later...

    K̂ = qˢ .* K .* qᵗ'
    k̂ = vec(sum(K̂, dims=1))
    K̂ .*= Zⁱ


    K̂ᵀZ = K̂'/(I - W)

    k̂diagZⁱ = k̂.*[Zⁱ[targetnodes[t], t] for t in 1:length(targetnodes)]

    Zrows = (I - W')\Matrix(sparse(targetnodes,
                                   1:length(targetnodes),
                                   1.0,
                                   size(W, 1),
                                   length(targetnodes)))
    k̂diagZⁱZ = k̂diagZⁱ .* Zrows'

    K̂ᵀZ_minus_diag = K̂ᵀZ - k̂diagZⁱZ

    edge_betweennesses = copy(W)

    for i in axes(W, 1)
        # ZᵀZⁱ_minus_diag = ZᵀKZⁱ[i,:] .- (k.*Z[targetnodes,i].*(Zⁱ[targetnodes,targetnodes]))'
        # ZᵀZⁱ_minus_diag = Z[:,i]'*K̂ .- (k.*Z[targetnodes,i].*diag(Zⁱ))'

        for j in findall(W[i,:].>0)
            edge_betweennesses[i,j] = W[i,j] .* (Z[j,:]'*K̂ᵀZ_minus_diag[:,i])[1]
        end
    end

    return edge_betweennesses
end




function RSP_expected_cost(W::SparseMatrixCSC,
                           C::SparseMatrixCSC,
                           Z::AbstractMatrix,
                           landmarks::AbstractVector)

    if axes(W) != axes(C)
        throw(DimensionMismatch(""))
    end
    if axes(Z, 1) != axes(W, 1)
        throw(DimensionMismatch(""))
    end
    if axes(Z, 2) != axes(landmarks, 1)
        Z = Z[:,landmarks]
    end

    if size(Z, 1) == size(Z, 2)
        C̄   = Z*((C .* W)*Z)
    else
        C̄   = (I - W)\((C .* W)*Z)
    end

    C̄ ./= Z
    # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
    replace!(C̄, NaN => Inf)
    dˢ  = [C̄[landmarks[j], j] for j in axes(Z, 2)]
    C̄ .-= dˢ'
    return C̄
end

RSP_free_energy_distance(Z::AbstractMatrix, θ::Real, landmarks::AbstractVector) =
    -log.(RSP_survival_probability(Z, θ, landmarks))./θ

RSP_survival_probability(Z::AbstractMatrix, θ::Real, landmarks::AbstractVector) =
    Z .* inv.([Z[i, j] for (j, i) in enumerate(landmarks)])'

RSP_power_mean_proximity(Z::AbstractMatrix, θ::Real, landmarks::AbstractVector) =
    RSP_survival_probability(Z, θ, landmarks).^(1/θ)

function connected_habitat(qˢ::AbstractVector, # Source qualities
                           qᵗ::AbstractVector, # Target qualities
                           S::AbstractMatrix)  # Matrix of proximities

    return qˢ .* (S*qᵗ)
end
