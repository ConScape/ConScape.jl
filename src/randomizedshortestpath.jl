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

    qˢZⁱqᵗ = qˢ .* Zⁱ .* qᵗ'

    edge_betweennesses = copy(W)

    sumqˢ = sum(qˢ)

    n = size(W,1)

    ZᵀZⁱ = Vector{Float64}(undef,n)

    for i in axes(W, 1)
        ZᵀZⁱ_minus_diag = Z[:,i]'*qˢZⁱqᵗ .- sumqˢ.* (Z[:,i].*diag(Zⁱ).*qᵗ)'

        i_idx = findall(W[i,:].>0)
        for j in i_idx

            edge_betweennesses[i,j] = W[i,j] .* (ZᵀZⁱ_minus_diag * Z[j,:])[1]
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

    KZⁱ = qˢ .* K .* qᵗ'
    k = vec(sum(KZⁱ, dims=1))

    qˢZⁱqᵗ = qˢ .* Zⁱ .* qᵗ'

    bet_matrix = copy(W)

    sumqˢ = sum(qˢ)

    n = size(W,1)

    ZᵀZⁱ = Vector{Float64}(undef,n)

    KZⁱ .*= Zⁱ

    for i in axes(W, 1)
        ZᵀZⁱ_minus_diag = Z[:,i]'*KZⁱ .- (k.*Z[:,i].*diag(Zⁱ))'

        i_idx = findall(W[i,:].>0)
        for j in i_idx

            bet_matrix[i,j] = W[i,j] .* (ZᵀZⁱ_minus_diag * Z[j,:])[1]
        end
    end

    return bet_matrix
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
        throw(DimensionMismatch(""))
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
    -log.(Z .* inv.([Z[i, j] for (j, i) in enumerate(landmarks)])')./β

function RSP_functionality(qˢ::AbstractVector, # Source qualities
                           qᵗ::AbstractVector, # Target qualities
                           S::AbstractMatrix)  # Matrix of similarities

    return qˢ .* (S*qᵗ)
end

function LF_sensitivity(qˢ::AbstractVector, # Source qualities
                        qᵗ::AbstractVector, # Target qualities
                        A::SparseMatrixCSC,
                        C::SparseMatrixCSC,
                        β::Real,
                        diff_C_A::SparseMatrixCSC,
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

    Nij_non = Matrix{Float64}(undef,n,n)
    Nij_hit = Matrix{Float64}(undef,n,n)
    Qij = Matrix{Float64}(undef,n,n)
    Γij = Matrix{Float64}(undef,n,n)

    pdiff_cij = Matrix{Float64}(undef,n,n)
    pdiff_aij = Matrix{Float64}(undef,n,n)

    diff_aij = Matrix{Float64}(undef,n,n)

    K_diff = Matrix{Float64}(undef,n,n)


    rowsums = sum(A,dims=2)


    CW  = C.*W
    ZCW = Z*CW
    ZCWZ = ZCW*Z

    EC = ZCWZ.*Zⁱ
    EC .-= diag(EC)'
    K = map(invcost, EC)
    K[1:n+1:end] .= 1
    K[isinf.(K)] .= 0

    S_e_cost = copy(A)
    S_e_aff = copy(A)


    # @showprogress for i = 1:n
    for i = 1:n
        Ni_non .= (Z[:,i].*Z[i,:]').*Zⁱ

        Ni_hit .= Ni_non .- diag(Ni_non)'

        Qi .= (ZCWZ[:,i].*Z[i,:]').*Zⁱ
        Qi .-= diag(Qi)'

        Γi_div .= (Ni_non.*EC .- Ni_hit.*EC[i,:]' .- Qi)./rowsums[i]

        i_idx = findall(A[i,:].>0)

        for j in i_idx
            Nij_non .= W[i,j] .* (Z[:,i].*Z[j,:]').*Zⁱ

            Nij_hit .= Nij_non .- diag(Nij_non)'

            Qij .= W[i,j] .* (ZCWZ[:,i].*Z[j,:]').*Zⁱ
            Qij .-= diag(Qij)'


            Γij .= Nij_non .* EC .- Nij_hit .* (EC[j,:] .+ C[i,j])' .- Qij

            pdiff_cij .= Nij_hit .+ β.*Γij
            K_diff .= K.*pdiff_cij
            S_e_cost[i,j] = -(qˢ'*K_diff)*qᵗ # A[i,j].*(-qˢ'*(K.*diff_aij)*qᵗ)

            pdiff_aij .= Γi_div .- Γij./A[i,j]
            K_diff .= K.*pdiff_aij
            S_e_aff[i,j] = -(qˢ'*K_diff)*qᵗ # A[i,j].*(-qˢ'*(K.*diff_aij)*qᵗ)


            # diff_cij = pdiff_cij + pdiff_aij.*diff_A_C[i,j]
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
    K ./= diag(Z)'
    K .^= inv(β) # \mathcal{Z}^{1/β}

    rowsums = sum(A,dims=2)

    bet_edge_k = RSP_edge_betweenness_kweighted(W, Z, qˢ, qᵗ, K, landmarks)
    S_e_cost = -bet_edge_k

    bet_node_k = RSP_betweenness_kweighted(W, Z, qˢ, qᵗ, K, landmarks)

    Idx = A.>0
    S_e_aff = copy(A)
    S_e_aff[Idx] = (bet_edge_k[Idx]./A[Idx] .- (Idx.*(bet_node_k./rowsums))[Idx])./β

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
