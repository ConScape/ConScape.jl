_Pref(A::SparseMatrixCSC) = Diagonal(inv.(vec(sum(A, dims=2)))) * A

function _W(Pref::SparseMatrixCSC, β::Real, C::SparseMatrixCSC)

    n = LinearAlgebra.checksquare(Pref)
    if LinearAlgebra.checksquare(C) != n
        throw(DimensionMismatch("Pref and C must have same size"))
    end

    W = Pref .* exp.((-).(β) .* C)
    replace!(W, NaN => 0.0)
    return W
end

function RSP_betweenness_qweighted(W::SparseMatrixCSC,
                                   Z::AbstractMatrix,
                                   qˢ::AbstractVector,
                                   qᵗ::AbstractVector,
                                   targetnodes::AbstractVector)

    Zⁱ = inv.(Z)
    Zⁱ[Z.==0] .= 1 # To prevent Inf*0 later...

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
                                   S::AbstractMatrix,  # Matrix of similarities
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
    Zⁱ[Z.==0] .= 1 # To prevent Inf*0 later...

    KZⁱ = qˢ .* S .* qᵗ'
    k = vec(sum(KZⁱ, dims=1))

    KZⁱ .*= Zⁱ
    for j in axis2
        KZⁱ[landmarks[j], j] -= k[j] .* Zⁱ[landmarks[j], j]
    end

    ZKZⁱt = (I - W)'\KZⁱ
    ZKZⁱt .*= Z

    return vec(sum(ZKZⁱt, dims=2)) # diag(Z * KZⁱ')
end

function RSP_edge_betweenness_qweighted(W::SparseMatrixCSC,
                                   Z::AbstractMatrix,
                                   qˢ::AbstractVector,
                                   qᵗ::AbstractVector,
                                   targetnodes::AbstractVector)

    Zⁱ = inv.(Z)
    Zⁱ[Z.==0] .= 1 # To prevent Inf*0 later...

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
                                        K::AbstractMatrix,  # Matrix of similarities
                                        targetnodes::AbstractVector)

    Zⁱ = inv.(Z)
    Zⁱ[Z.==0] .= 1 # To prevent Inf*0 later...

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




function RSP_dissimilarities(W::SparseMatrixCSC,
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

RSP_free_energy_distance(Z::AbstractMatrix, β::Real, landmarks::AbstractVector) =
    -log.(RSP_survival_probability(Z, β, landmarks))./β

RSP_survival_probability(Z::AbstractMatrix, β::Real, landmarks::AbstractVector) =
    Z .* inv.([Z[i, j] for (j, i) in enumerate(landmarks)])'

RSP_power_mean_proximity(Z::AbstractMatrix, β::Real, landmarks::AbstractVector) =
    RSP_survival_probability(Z, β, landmarks).^(1/β)

function RSP_functionality(qˢ::AbstractVector, # Source qualities
                           qᵗ::AbstractVector, # Target qualities
                           S::AbstractMatrix)  # Matrix of similarities

    return qˢ .* (S*qᵗ)
end



function LF_sensitivity(A::SparseMatrixCSC,
                                   C::SparseMatrixCSC,
                                   β::Real,
                                   W::SparseMatrixCSC,
                                   Z::AbstractMatrix,
                                   K::AbstractMatrix,  # diff_K_D (but can be any weighting matrix)
                                   qˢ::AbstractVector,
                                   qᵗ::AbstractVector,
                                   lmarks::AbstractVector)

    Zⁱ = inv.(Z)
    Zⁱ[Z.==0] .= 1 # To prevent Inf*0 later...

    K̂ = qˢ .* K .* qᵗ'
    k̂ = vec(sum(K̂, dims=1))
    K̂ .*= Zⁱ # k̂ᵢⱼ = kᵢⱼ/zᵢⱼ

    CW = C.*W

    Q = (I-W)\(CW*Z)

    C̄ᵣ = Q.*Zⁱ # Expected costs of REGULAR paths

    K̂ᵀZ = K̂'/(I - W)

    k̂diagZⁱ = k̂.*[Zⁱ[lmarks[t], t] for t in 1:length(lmarks)]

    if size(Z,2) < size(Z,1)
        I_L = Matrix(sparse(lmarks, 1:length(lmarks), 1.0, size(W, 1), length(lmarks)))
        Zrows = I_L'/(I - W)
    else
        Zrows = Z
    end

    X3 = k̂diagZⁱ .* Zrows

    k̂diagC̄Zⁱ = k̂diagZⁱ.*[C̄ᵣ[lmarks[t], t] for t in 1:length(lmarks)]
    X1 = ((K̂.*C̄ᵣ)' - (K̂ᵀZ*CW) + (X3*CW))/(I-W) - k̂diagC̄Zⁱ .* Zrows # "X1- X2 - X4"

    X3 .= K̂ᵀZ .- X3

    kΣ = copy(W) # k-weighted negative covariance matrix
    kB = copy(W) # k-weighted edge betweenness matrix

    @showprogress 3 for i in axes(W, 1)
        for j in findall(W[i,:].>0)
            kB[i,j] *= (Z[j,:]'*X3[:,i])[1]
            kΣ[i,j] *= (Z[j,:]'*X1[:,i])[1] - (Q[j,:]'*X3[:,i])[1] - C[i,j]*kB[i,j]/W[i,j]
        end
    end

    # kB .*= W
    # kΣ .*= W

    kΣ_node = sum(kΣ, dims=2)
    Ae = sum(A,dims=2)

    S_cost = kB + β*kΣ

    Idx = W.>0
    Aⁱ = mapnz(x -> inv(x), A)
    S_aff = (kΣ_node./Ae).*Idx - kΣ.*Aⁱ

    return S_aff, S_cost
end



function LF_sensitivity_old(qˢ::AbstractVector, # Source qualities
                        qᵗ::AbstractVector, # Target qualities
                        A::SparseMatrixCSC,
                        C::SparseMatrixCSC,
                        β::Real,
                        diff_C_A::SparseMatrixCSC,
                        diff_K_D::AbstractMatrix,
                        W::SparseMatrixCSC,
                        Z::AbstractMatrix,
                        invcost::Any,
                        landmarks::AbstractVector)

    n = size(A,1)
    Zⁱ = inv.(Z)
    Zⁱ[Z.==0] .= 1 # To prevent Inf*0 later...

    # Preallocations:
    Ni_non = Matrix{Float64}(undef,n,n)
    Ni_hit = Matrix{Float64}(undef,n,n)
    Qi = Matrix{Float64}(undef,n,n)
    Γi_div = Matrix{Float64}(undef,n,n)
    EC_non_minus_i = Matrix{Float64}(undef,n,n)

    Nij_non = Matrix{Float64}(undef,n,n)
    Nij_hit = Matrix{Float64}(undef,n,n)
    Qij = Matrix{Float64}(undef,n,n)
    Γij = Matrix{Float64}(undef,n,n)

    pdiff_cij = Matrix{Float64}(undef,n,n)
    pdiff_aij = Matrix{Float64}(undef,n,n)


    rowsums = sum(A,dims=2)


    # CW  = C.*W
    # ZCW = Z*CW
    # ZCWZ = ZCW*Z
    # EC_non = ZCWZ.*Zⁱ


    EC_non = (Z*(C.*W)*Z).*Zⁱ
    EC = EC_non .- diag(EC_non)'

    S_e_cost = copy(A)
    S_e_aff = copy(A)


    @showprogress 3 for i = 1:n # Shows progress bar if process takes more than 3 secs
    # for i = 1:n
        Ni_non .= (Z[:,i].*Z[i,:]').*Zⁱ

        Ni_hit .= Ni_non .- diag(Ni_non)'


        EC_non_minus_i .= EC_non.-EC_non[:,i]
        Qi .= Ni_non.*EC_non_minus_i
        Qi .-= diag(Qi)'

        Γi_div .= (Ni_hit.*EC_non[i,:]' .- Qi)./rowsums[i]

        i_idx = findall(A[i,:].>0)

        # TODO: Consider only nonzero quality target nodes!
        # for j in i_idx ∩ target_idx
        for j in i_idx
            Nij_non .= W[i,j] .* (Z[:,i].*Z[j,:]').*Zⁱ

            Nij_hit .= Nij_non .- diag(Nij_non)'

            Qij .= Nij_non.*EC_non_minus_i
            Qij .-= diag(Qij)'

            Γij .= Nij_hit .* (EC_non[j,:] .+ C[i,j])' .- Qij

            pdiff_cij .= Nij_hit .- β.*Γij
            # pdiff_cij[j,:] .= 0
            # pdiff_cij[:,j] .= 0
            S_e_cost[i,j] = (qˢ'* (diff_K_D.*pdiff_cij)) *qᵗ # A[i,j].*(-qˢ'*(K.*diff_aij)*qᵗ)

            pdiff_aij .= Γij./A[i,j] .- Γi_div
            # pdiff_aij[j,:] .= 0
            # pdiff_aij[:,j] .= 0
            S_e_aff[i,j] = (qˢ'* (diff_K_D.*pdiff_aij)) *qᵗ # A[i,j].*(-qˢ'*(K.*diff_aij)*qᵗ)


            # diff_cij = pdiff_cij + pdiff_aij.*diff_C_A[i,j]
            # diff_cij[:,i] .= 0
            # S_e_total[i,j] = -(qˢ'*K_diff)*qᵗ # A[i,j].*(-qˢ'*(K.*diff_aij)*qᵗ)

        end

    end

    S_e_total = S_e_aff .+ S_e_cost.*diff_C_A

    return S_e_total, S_e_aff, S_e_cost

end










function LF_power_mean_sensitivity(qˢ::AbstractVector, # Source qualities
                                   qᵗ::AbstractVector, # Target qualities
                                   A::SparseMatrixCSC,
                                   β::Real,
                                   diff_C_A::SparseMatrixCSC,
                                   W::SparseMatrixCSC,
                                   Z::AbstractMatrix,
                                   landmarks::AbstractVector)

    K = copy(Z)
    K ./= [Z[landmarks[i],i] for i in 1:length(landmarks)]'
    K .^= inv(β) # \mathcal{Z}^{1/β}

    rowsums = sum(A,dims=2)

    bet_edge_k = RSP_edge_betweenness_kweighted(W, Z, qˢ, qᵗ, K, landmarks)
    S_e_cost = -bet_edge_k

    bet_node_k = RSP_betweenness_kweighted(W, Z, qˢ, qᵗ, K, landmarks)

    Idx = A.>0
    Aⁱ = ConScape.mapnz(inv, A)
    S_e_aff = (bet_edge_k.*Aⁱ .- (bet_node_k./rowsums).*Idx)./β

    return S_e_aff .+ S_e_cost.*diff_C_A, S_e_aff, S_e_cost

end



# Compute a (column subset of a) dense identity matrix where the subset corresponds
# to the landsmarks
function _Imn(n::Integer, landmarks::AbstractVector)
    Imn = zeros(n, length(landmarks))
    for (j, i) in enumerate(landmarks)
        Imn[i, j] = 1
    end
    return Imn
end
