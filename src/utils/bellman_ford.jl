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
    rawDistances = Graphs.dijkstra_shortest_paths(SimpleWeightedDiGraph(C), target).dists
    idx = sortperm(rawDistances) # Compute the update order (topological sorting)
    rawDistances = rawDistances[idx]
    convergence = false
    iter = 0

    trPref = copy(Pref')
    trC = copy(C')

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
                    if rawDistances[node-1] == rawDistances[node]
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
        if iter == 1
            continue
        end
        # check if the free energy and the RSP have converged
        convergence = (maximum(abs, φ - φ_1) / maximum(φ) < 1e-8) & (maximum(abs, c̄ - c̄_1) / maximum(c̄) < 1e-8)
    end
    return c̄, φ
end

# Updates the RSP and free energy vectors for a given list of nodes
# Inputs:
#   c̄: the directed expected cost (RSP dissimilarity)
#   φ: the directed free energy
#   trPref: the (transposed) transition probability matrix
#   trC: the (transposed) cost matrix
#   θ: the inverse temperature
#   updatelist: the list of nodes that should be updated simultaneously
# Outputs:
#   c̄: the updated directed expected cost (RSP dissimilarity)
#   φ: the updated directed free energy
# Comment:
#   The two sparse arrays in passed in transposed form since it makes the access much more efficient
function _bellman_ford_update_transposed!(c̄::Vector, φ::Vector, trPref::SparseMatrixCSC, trC::SparseMatrixCSC, θ::Real, updatelist::Vector)
    if length(updatelist) == 1
        index = updatelist[1]
        ec, v = _bellman_ford_update_node_transposed(c̄, φ, trPref, trC, θ, index)
        c̄[index] = ec
        φ[index] = v
        return c̄, φ
    end
    prev_φ = copy(φ)
    prev_c̄ = copy(c̄)
    for i in 1:length(updatelist)
        index = updatelist[i]
        ec, v = _bellman_ford_update_node_transposed(prev_c̄, prev_φ, trPref, trC, θ, index)
        c̄[index] = ec
        φ[index] = v
    end
    return c̄, φ
end

# Helper function required for good performance until https://github.com/JuliaLang/julia/pull/42647 has been released
function fast_getindex(A::SparseMatrixCSC{Tv,Ti}, I::AbstractVector, J::Integer) where {Tv,Ti}
    if !issorted(I)
        throw(ArgumentError("only sorted indices are currectly supported"))
    end

    nI = length(I)

    nzind = Ti[]
    nzval = Tv[]

    iI = 1
    for iptr in A.colptr[J]:(A.colptr[J+1]-1)
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
#   c̄: the directed expected cost (RSP dissimilarity)
#   φ: the directed free energy
#   trPref: the (transposed) transition probability matrix
#   trC: the (transposed) cost matrix
#   θ: the inverse temperature
#   index: the index of the node that should be updated
# Outputs:
#   ec: the updated directed expected cost (RSP dissimilarity) for the node
#   v: the updated directed free energy for the node
# Comment:
#   The two sparse arrays in passed in transposed form since it makes the access much more efficient
function _bellman_ford_update_node_transposed(c̄::Vector, φ::Vector, trPref::SparseMatrixCSC, trC::SparseMatrixCSC, θ::Real, index::Integer)
    Prefindex = trPref[:, index]
    idx = Prefindex.nzind # Get the list of successors
    # computation of θ(cᵢⱼ+φ(j,t))-log([Pʳᵉᶠ]ᵢⱼ)
    # ect = (Array(trC[idx, index]) + φ[idx]) .* θ .- log.(Prefindex.nzval)
    ect = (Array(fast_getindex(trC, idx, index)) .+ φ[idx]) .* θ .- log.(Prefindex.nzval)
    finiteidx = isfinite.(ect)
    idx = idx[finiteidx]
    ect = ect[finiteidx]

    if isempty(idx)
        throw(ErrorException("the node $index has no valid successor in the DAG"))
    end

    # First check there is only one neighbor, if so, the solution is trivial
    if length(idx) == 1
        return c̄[idx[1]] + trC[idx[1], index], ect[1] / θ
    end

    # log-sum-exp trick
    minval = minimum(ect) # computation of cᵢ*
    ect .-= minval # remove the lowest value from all the vector
    v = (minval - log(sum(exp, -ect))) / θ # computation of the directed free energy
    if isinf(v)
        throw(ErrorException("infinite valude in the distance vector at index $index"))
    end

    #computation of the updated expected cost based on the free energy
    ec = zero(eltype(c̄))
    for j in 1:length(idx)
        trCidxjindex = trC[idx[j], index]
        pij = trPref[idx[j], index] * exp(θ * (v - φ[idx[j]] - trCidxjindex))
        ec += pij * (trCidxjindex + c̄[idx[j]])
    end
    return ec, v
end