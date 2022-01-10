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

# Returns the directed RSP dissimilarity and directed free energy distance for all nodes to a given target
# Inputs:
# Pref: the transition probability matrix
# C: the cost matrix
# θ: the inverse temperature
# target: the index of the target node
# approx: a boolean value, with:
#     true: computes the single-pass approximation
#     false: loops until convergence
# Outputs:
# c̄: The directed RSP dissimilarity (or approximation thereof) from all nodes to the target
# φ: The directed free energy distance (or approximation thereof) from all nodes to the target
function bellman_ford(Pref::SparseMatrixCSC, C::SparseMatrixCSC, θ::Real, target::Integer, approx::Bool)
    n = LinearAlgebra.checksquare(C)
    if LinearAlgebra.checksquare(Pref) != n
        throw(DimensionMismatch("the dimensions of the matrices Pref and C do not match"))
    end
    if target < 1 || target > n
        throw(DimensionMismatch("the target node is not valid"))
    end
    if θ <= 0
        throw(ArgumentError("the value of θ must be strictly positive"))
    end

    # Initialize the Free Energy and RSP vector
    φ = ones(n)
    φ[target] = 0
    replace!(φ, 1 => Inf) # Andreas: why?
    c̄ = copy(φ)

    # Compute the raw distances to the target for the DAG (in out case, the least-cost)
    rawDistances = dijkstra_shortest_paths(SimpleWeightedDiGraph(C), target).dists
    idx = sortperm(rawDistances) # Compute the update order (topological sorting)
    rawDistances = rawDistances[idx]
    convergence = false
    iter = 0

    trPref = copy(Pref')
    trC    = copy(C')

    while !convergence
        φ_1 = copy(φ)
        c̄_1 = copy(c̄)
        updatelist = [-1] # The updatelist contains the list of nodes that should be updated simultaneously
        for node in 1:n
            index = idx[node]
            if index != target
                if updatelist[1] == -1
                    updatelist = [index]
                else
                    if rawDistances[node - 1] == rawDistances[node]
                        append!(updatelist, index) # Equidistant nodes should be updated simultaneously
                    else
                        c̄, φ = _bellman_ford_update_transposed!(c̄, φ, trPref, trC, θ, updatelist)
                        updatelist = [index]
                    end
                    if node == n
                        c̄, φ = _bellman_ford_update_transposed!(c̄, φ, trPref, trC, θ, updatelist)
                    end
                end
            end
        end
        if approx
            break # Break the loop if in a single pass approach
        end
        iter += 1
        if iter==1
            continue
        end
        # check if the free energy and the RSP have converged
        convergence=(maximum(abs, φ - φ_1)/maximum(φ) < 1e-8) & (maximum(abs, c̄ - c̄_1)/maximum(c̄) < 1e-8)
    end
    return c̄, φ
end

# Updates the RSP and free energy vectors for a given list of nodes
# Inputs:
    # c̄: the directed expected cost (RSP dissimilarity)
    # φ: the directed free energy
    # trPref: the (transposed) transition probability matrix
    # trC: the (transposed) cost matrix
    # θ: the inverse temperature
    # updatelist: the list of nodes that should be updated simultaneously
# Outputs:
    # c̄: the updated directed expected cost (RSP dissimilarity)
    # φ: the updated directed free energy
# Comment:
    # The two sparse arrays in passed in transposed form since it makes the access much more efficient
function _bellman_ford_update_transposed!(c̄::Vector, φ::Vector, trPref::SparseMatrixCSC, trC::SparseMatrixCSC, θ::Real, updatelist::Vector)
    if length(updatelist) == 1
        index = updatelist[1]
        ec, v = _bellman_ford_update_node_transposed(c̄, φ, trPref, trC, θ, index)
        c̄[index] = ec
        φ[index] = v
        return c̄, φ
    end
    prev_φ=copy(φ)
    prev_c̄=copy(c̄)
    for i in 1:length(updatelist)
        index = updatelist[i]
        ec, v = _bellman_ford_update_node_transposed(prev_c̄, prev_φ, trPref, trC, θ, index)
        c̄[index] = ec
        φ[index] = v
    end
    return c̄, φ
end

# Helper function required for good performance until https://github.com/JuliaLang/julia/pull/42647 has been released
function mygetindex(A::SparseMatrixCSC{Tv,Ti}, I::AbstractVector, J::Integer) where {Tv,Ti}
    if !issorted(I)
        throw(ArgumentError("only sorted indices are currectly supported"))
    end

    nI = length(I)

    nzind = Ti[]
    nzval = Tv[]

    iI = 1
    for iptr in A.colptr[J]:(A.colptr[J + 1] - 1)
        iA = A.rowval[iptr]
        while iI <= nI && I[iI] <= iA
            if I[iI] == iA
                push!(nzval, A.nzval[iptr])
                push!(nzind, iI)
            end
            iI += 1
        end
    end
    return SparseVector(length(I), nzind, nzval)
end

# Updates the directed RSP and direct free energy value for a given node
# Inputs:
    # c̄: the directed expected cost (RSP dissimilarity)
    # φ: the directed free energy
    # trPref: the (transposed) transition probability matrix
    # trC: the (transposed) cost matrix
    # θ: the inverse temperature
    # index: the index of the node that should be updated
# Outputs:
    # ec: the updated directed expected cost (RSP dissimilarity) for the node
    # v: the updated directed free energy for the node
# Comment:
    # The two sparse arrays in passed in transposed form since it makes the access much more efficient
function _bellman_ford_update_node_transposed(c̄::Vector, φ::Vector, trPref::SparseMatrixCSC, trC::SparseMatrixCSC, θ::Real, index::Integer)
    Prefindex = trPref[:,index]
    idx = Prefindex.nzind # Get the list of successors
    # computation of θ(cᵢⱼ+φ(j,t))-log([Pʳᵉᶠ]ᵢⱼ)
    # ect = (Array(trC[idx, index]) + φ[idx]) .* θ .- log.(Prefindex.nzval)
    ect = (Array(mygetindex(trC, idx, index)) .+ φ[idx]) .* θ .- log.(Prefindex.nzval)
    finiteidx = isfinite.(ect)
    idx = idx[finiteidx]
    ect = ect[finiteidx]

    if isempty(idx)
        throw(ErrorException("the node $index has no valid successor in the DAG"))
    end

    # First check there is only one neighbor, if so, the solution is trivial
    if length(idx) == 1
        return c̄[idx[1]] + trC[idx[1], index], ect[1]/θ
    end

    # log-sum-exp trick
    minval = minimum(ect) # computation of cᵢ*
    ect .-= minval # remove the lowest value from all the vector
    v = (minval - log(sum(exp, -ect)))/θ # computation of the directed free energy
    if isinf(v)
        throw(ErrorException("infinite valude in the distance vector at index $index"))
    end

    #computation of the updated expected cost based on the free energy
    ec = zero(eltype(c̄))
    for j in 1:length(idx)
        trCidxjindex = trC[idx[j], index]
        pij = trPref[idx[j], index]*exp(θ*(v - φ[idx[j]] - trCidxjindex))
        ec += pij*(trCidxjindex + c̄[idx[j]])
    end
    return ec, v
end


function LF_sensitivity(A::SparseMatrixCSC,
                        C::SparseMatrixCSC,
                        θ::Real,
                        W::SparseMatrixCSC,
                        Z::AbstractMatrix,
                        K::AbstractMatrix,  # diff_K_D (but can be any weighting matrix)
                        qˢ::AbstractVector,
                        qᵗ::AbstractVector,
                        lmarks::AbstractVector)

    Zⁱ = inv.(Z)
    Zⁱ[.!isfinite.(Zⁱ)] .= floatmax(eltype(Z)) # To prevent Inf*0 later...

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
    Ae = sum(A, dims=2)

    S_cost = kB + kΣ/θ

    Idx = W.>0
    Aⁱ = mapnz(x -> inv(x), A)
    S_aff = (kΣ_node./Ae).*Idx - kΣ.*Aⁱ

    return S_aff, S_cost
end


function LF_power_mean_sensitivity(qˢ::AbstractVector, # Source qualities
                                   qᵗ::AbstractVector, # Target qualities
                                   A::SparseMatrixCSC,
                                   θ::Real,
                                   diff_C_A::SparseMatrixCSC,
                                   W::SparseMatrixCSC,
                                   Z::AbstractMatrix,
                                   landmarks::AbstractVector)

    K = copy(Z)
    K ./= [Z[landmarks[i],i] for i in 1:length(landmarks)]'
    K .^= θ # \mathcal{Z}^θ

    rowsums = sum(A,dims=2)

    bet_edge_k = RSP_edge_betweenness_kweighted(W, Z, qˢ, qᵗ, K, landmarks)
    S_e_cost = -bet_edge_k

    bet_node_k = RSP_betweenness_kweighted(W, Z, qˢ, qᵗ, K, landmarks)

    Idx = A.>0
    Aⁱ = ConScape.mapnz(inv, A)
    S_e_aff = (bet_edge_k.*Aⁱ .- (bet_node_k./rowsums).*Idx).*θ

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
