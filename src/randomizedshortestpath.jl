# Generate the sparse diagonal rhs matrix
function sparse_rhs(targetnodes, n)
    sparse(targetnodes,
        1:length(targetnodes),
        1.0,
        n,
        length(targetnodes),
    )
end

_inv(Z) = _inv!(similar(Z), Z)
function _inv!(Zⁱ, Z)
    broadcast(Z) do x
        x = inv(x)
        isfinite(x) ? x : floatmax(eltype(Z))
    end
end

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
                                   targetnodes::AbstractVector;
    Zⁱ=_inv(Z),
    workspaces=[similar(Z), similar(Z)],
    solver=nothing,
    Aadj = (I - W)',
    Aadj_init=init(solver, Aadj),
    kw...
)
    workspace1, workspace2 = workspaces

    qˢZⁱqᵗ = workspace1
    qˢZⁱqᵗ .= qˢ .* Zⁱ .* qᵗ'
    sumqˢ = sum(qˢ)
    for j in axes(Z, 2)
        qˢZⁱqᵗ[targetnodes[j], j] -=  sumqˢ * qᵗ[j] * Zⁱ[targetnodes[j], j]
    end

    # TODO adjoint of LinearSolver?
    ZqˢZⁱqᵗZt = ldiv!(solver, Aadj_init, qˢZⁱqᵗ; B_copy=copy!(workspace2, qˢZⁱqᵗ))
    ZqˢZⁱqᵗZt .*= Z

    return sum(ZqˢZⁱqᵗZt, dims=2) # diag(Z * ZqˢZⁱqᵗ')
end


function RSP_betweenness_kweighted(W::SparseMatrixCSC,
                                   Z::AbstractMatrix,  # Fundamental matrix of non-absorbing paths
                                   qˢ::AbstractVector, # Source qualities
                                   qᵗ::AbstractVector, # Target qualities
                                   S::AbstractMatrix,  # Matrix of proximities
                                   landmarks::AbstractVector;
    Zⁱ=_inv(Z),
    workspaces=[similar(Z)],
    solver=nothing,
    Aadj=(I - W)',
    Aadj_init=init(solver, Aadj),
    kw...
)
    workspace1 = workspaces[1]
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

    # Write into proximities
    KZⁱ = S
    KZⁱ .*= qˢ .* qᵗ'

    # If any of the values of KZⁱ is above one then there is a risk of overflow.
    # Hence, we scale the matrix and apply the scale factor by the end of the
    # computation.
    λ = max(1.0, maximum(KZⁱ))
    # k = vec(sum(KZⁱ, dims=1)) * inv(λ)
    ws_col = view(workspace1, 1:1, :)
    k = vec(sum!(ws_col, KZⁱ))
    k .*= inv(λ)

    KZⁱ .*= inv.(λ) .* Zⁱ
    for j in axis2
        KZⁱ[landmarks[j], j] -= k[j] .* Zⁱ[landmarks[j], j]
    end

    # KZi overwritten from here
    # ZKZⁱt = (I - W)'\KZⁱ
    ZKZⁱt = ldiv!(solver, Aadj_init, KZⁱ; B_copy=copy!(workspace1, KZⁱ))
    ZKZⁱt .*= λ .* Z

    scratch = view(workspace1, :, 1:1)
    return vec(sum!(scratch, ZKZⁱt)) # diag(Z * KZⁱ')
    # return vec(sum(ZKZⁱt, dims=2)) # diag(Z * KZⁱ')
end

function RSP_edge_betweenness_qweighted(W::SparseMatrixCSC,
                                        Z::AbstractMatrix,
                                        qˢ::AbstractVector,
                                        qᵗ::AbstractVector,
                                        targetnodes::AbstractVector;
    solver=nothing,
    Zⁱ=_inv(Z),
    workspaces=[similar(Z), similar(Z), similar(Z)],
    Aadj=(I - W)',
    Aadj_init=init(solver, Aadj),
    B_sparse=sparse_rhs(targetnodes, size(W, 1)),
    kw...
)
    edge_betweennesses = copy(W)
    n = size(W, 1)
    workspace1, workspace2, workspace3 = workspaces

    diagZⁱ = [Zⁱ[targetnodes[t], t] for t in 1:length(targetnodes)]
    sumqˢ = sum(qˢ)

    # FIXME: This should be only done when actually size(Z, 2) < size(Z, 1)/K where K ≈ 10 or so.
    # Otherwise we just compute many of the elements of Z twice...
    if size(Z, 2) < size(Z, 1)
        B = workspace1 .= B_sparse
        Zrows = ldiv!(solver, Aadj_init, B; B_copy=copy!(workspace2, B))'
        Zrows .*= sumqˢ * qᵗ .* diagZⁱ
    else
        Zrows = workspace1 .= Z .* (sumqˢ * qᵗ .* diagZⁱ)
    end

    qˢZⁱqᵗ = workspace2 .= qˢ .* Zⁱ .* qᵗ'
    # QZⁱᵀZ = qˢZⁱqᵗ' / A
    QZⁱᵀZ = ldiv!(solver, Aadj_init, qˢZⁱqᵗ; B_copy=copy!(workspace3, qˢZⁱqᵗ))'

    RHS = workspace3 .= QZⁱᵀZ .- Zrows

    for i in axes(W, 1)
        # ZᵀZⁱ_minus_diag = Z[:,i]'*qˢZⁱqᵗ .- sumqˢ.* (Z[:,i].*diag(Zⁱ).*qᵗ)'

        for (j, x) in enumerate(view(W, i, :))
            x > 0 || continue   
            # edge_betweennesses[i,j] = W[i,j] .* Zqt[j,:]'* (ZᵀZⁱ_minus_diag * Z[j,:])[1]
            edge_betweennesses[i, j] = W[i, j] .* (view(Z, j, :)' * view(RHS, :, i))[1]
        end
    end

    return edge_betweennesses
end

function RSP_edge_betweenness_kweighted(W::SparseMatrixCSC,
                                        Z::AbstractMatrix,
                                        qˢ::AbstractVector,
                                        qᵗ::AbstractVector,
                                        K::AbstractMatrix,  # Matrix of proximities
                                        targetnodes::AbstractVector;
    solver=nothing,
    workspaces=[similar(Z), similar(Z)],
    permuted_workspaces=(similar(Z'),),
    Zⁱ=_inv(Z),
    Aadj=(I - W)',
    Aadj_init=init(solver, Aadj),
    B_sparse=sparse_rhs(targetnodes, size(W, 1)),
    kw...
)
    edge_betweennesses = copy(W)
    workspace1, workspace2 = workspaces
    permuted_workspace1 = permuted_workspaces[1]

    K̂ = K .= qˢ .* K .* qᵗ'
    k̂ = vec(sum(K̂, dims=1))
    K̂ .*= Zⁱ

    # K̂ᵀZ =  K̂' / A # is equivalent to the below
    K̂ᵀZ = ldiv!(solver, Aadj_init, K̂; B_copy=copy!(workspace2, K̂))'

    k̂diagZⁱ = k̂ .* [Zⁱ[targetnodes[t], t] for t in 1:length(targetnodes)]

    B = workspace1 .= B_sparse
    Zrows = ldiv!(solver, Aadj_init, B; B_copy=copy!(workspace2, B))
    k̂diagZⁱZ = permuted_workspace1 .= k̂diagZⁱ .* Zrows'
    K̂ᵀZ_minus_diag = k̂diagZⁱZ .= K̂ᵀZ .- k̂diagZⁱZ

    for i in axes(W, 1)
        # ZᵀZⁱ_minus_diag = ZᵀKZⁱ[i,:] .- (k.*Z[targetnodes,i].*(Zⁱ[targetnodes,targetnodes]))'
        # ZᵀZⁱ_minus_diag = Z[:,i]'*K̂ .- (k.*Z[targetnodes,i].*diag(Zⁱ))'

        for (j, x) in enumerate(view(W, i, :))
            x > 0 || continue   
            edge_betweennesses[i, j] = W[i, j] .* (view(Z, j, :)' * view(K̂ᵀZ_minus_diag, :, i))[1]
        end
    end

    return edge_betweennesses
end


function RSP_expected_cost(W::SparseMatrixCSC,
                           C::SparseMatrixCSC,
                           Z::AbstractMatrix,
                           landmarks::AbstractVector;
    solver=nothing,
    A=(I - W),
    A_init=init(solver, A),
    workspaces=[similar(Z), similar(Z)],
    expected_costs=similar(Z),
    kw...
)
    CW = C .* W
    workspace1, workspace2 = workspaces
    if axes(W) != axes(C)
        throw(DimensionMismatch(""))
    end
    if axes(Z, 1) != axes(W, 1)
        throw(DimensionMismatch(""))
    end
    if axes(Z, 2) != axes(landmarks, 1)
        Z = Z[:,landmarks]
    end


    # When threaded the solver is faster than a dense matmul
    # C̄ = if size(Z, 1) == size(Z, 2)
        # B = mul!(workspace1, C .* W,  Z)
        # mul!(B, C .* W,  Z)
        # This is a dense-dense matmul... very slow
        # Z * B
    # else
        # TODO permuted workspace here for the broadcast
        B = mul!(workspace1, CW, Z)
        C̄ = ldiv!(solver, A_init, B; B_copy=copy!(workspace2, B))
    # end

    C̄ ./= Z
    # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
    replace!(C̄, NaN => Inf)
    dˢ = view(workspace2, 1, :)
    # TODO clarify what this does

    for j in axes(Z, 2)
        dˢ[j] = C̄[landmarks[j], j] 
    end
    C̄ .-= dˢ'
    return copyto!(expected_costs, C̄)
end

function RSP_free_energy_distance(Z::AbstractMatrix, θ::Real, landmarks::AbstractVector; 
    survival_probability=nothing, 
    free_energy_distances=similar(Z),
    kw...
)
    if isnothing(survival_probability)
        survival_probability = RSP_survival_probability(Z, θ, landmarks; kw...)
    end
    free_energy_distances .= -log.(max.(zero(eltype(Z)), survival_probability)) ./ θ

    return free_energy_distances
end

function RSP_survival_probability(Z::AbstractMatrix, θ::Real, landmarks::AbstractVector; kw...)
    Z .* inv.([Z[i, j] for (j, i) in enumerate(landmarks)])'
end

function RSP_power_mean_proximity(Z::AbstractMatrix, θ::Real, landmarks::AbstractVector; 
    survival_probability=nothing, kw...
)
    survival_probability = if isnothing(survival_probability) 
        RSP_survival_probability(Z, θ, landmarks; kw...)
    else
        survival_probability
    end
    survival_probability .^ (1 / θ)
end

function connected_habitat(qˢ::AbstractVector, # Source qualities
                           qᵗ::AbstractVector, # Target qualities
                           S::AbstractMatrix; # Matrix of proximities
    workspaces=[similar(S, size(S, 1), 1)],
    kw...
)  
    mul!(view(workspaces[1], :, 1), S, qᵗ) .*= qˢ
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
        convergence = (maximum(abs, φ - φ_1) / maximum(φ) < 1e-8) & (maximum(abs, c̄ - c̄_1) / maximum(c̄) < 1e-8)
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
