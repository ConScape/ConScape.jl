module ConScape

    using SparseArrays, Plots, LightGraphs, SimpleWeightedGraphs
    using LinearAlgebra

    mutable struct Grid
        shape::Tuple{Int,Int}
        A::SparseMatrixCSC{Float64,Int}
        id_to_grid_coordinate_list::Vector{Tuple{Int,Int}}
        source_qualities::Matrix{Float64}
        target_qualities::Matrix{Float64}
    end

    """
        Grid(;shape=nothing,
             source_qualities::Matrix=ones(shape),
             target_qualities::Matrix=ones(shape),
             nhood_size::Integer=8,
             landscape=_generateA(shape..., nhood_size)) -> Grid

    Construct a `Grid` from a `landscape` passed a `SparseMatrixCSC`.
    """
    function Grid(;shape=nothing,
                  qualities::Matrix=ones(shape),
                  source_qualities::Matrix=qualities,
                  target_qualities::Matrix=qualities,
                  nhood_size::Integer=8,
                  landscape=_generateA(shape..., nhood_size))
        @assert prod(shape) == LinearAlgebra.checksquare(landscape)
        N_grid = prod(shape)
        n_cols = shape[2]
        Grid(shape,
             landscape,
             _id_to_grid_coordinate_list(N_grid, n_cols),
             source_qualities,
             target_qualities)
    end

    Base.size(g::Grid) = g.shape

    # simulate a permeable wall
    function perm_wall_sim(;shape=(30,60),
                            Q=1,
                            A=0.5,
                            ww=3,
                            wp=0.5,
                            cw=(3,3),
                            cp=(0.35,0.7),
                            kwargs...)

        # 1. initialize landscape
        n_rows, n_cols = shape
        N = n_rows*n_cols
        g = Grid(; shape=shape, kwargs...)
        g.A = A * g.A

        # # 2. compute the wall
        wpt = round(Int, n_cols*wp - ww/2 + 1)
        xs  = range(wpt, stop=wpt + ww - 1)

        # 3. compute the corridors
        ys = Int[]
        for i in 1:length(cw)
            cpt = floor(Int, n_rows*cp[i]) - ceil(Int, cw[i]/2)
            if i == 1
                append!(ys, 1:cpt)
            else
                append!(ys, range(maximum(ys) + 1 + cw[i-1], stop=cpt))
                append!(ys, range(maximum(ys) + 1 + cw[i], stop=n_rows))
            end
        end

        impossible_nodes = vec(collect(Iterators.product(ys,xs)))
        _set_impossible_nodes!(g, impossible_nodes)
        # return [g]
        return g
    end

    #=
    Generate the affinity matrix of a grid graph, where each
    pixel is connected to its vertical and horizontal neighbors.

    Parameters:
    - nhood_size: 4 creates horizontal and vertical edges, 8 creates also diagonal edges
    =#
    function _generateA(n_rows, n_cols, nhood_size)

        N = n_rows*n_cols

        # A = ss.dok_matrix(N, N)
        is, js, vs = Int[], Int[], Float64[]
        for i in 1:n_rows
            for j in 1:n_cols
                n = (i - 1)*n_cols + j # current pixel
                if j < n_cols
                    # Add horizontal edge:
                    # A[n, n + 1] = 1
                    push!(is, n)
                    push!(js, n + 1)
                    push!(vs, 1)
                end
                if i < n_rows
                    # Add vertical edge:
                    # A[n, n + n_cols] = 1
                    push!(is, n)
                    push!(js, n + n_cols)
                    push!(vs, 1)

                    # TODO: WRITE THIS TO ALLOW OTHER VALUES OF nhood_size!
                    if nhood_size == 8
                        if j < n_cols
                            # Add lower-right diagonal edge:
                            # A[n, n + n_cols + 1] = 1 / √2
                            push!(is, n)
                            push!(js, n + n_cols + 1)
                            push!(vs, 1 / √2)
                        end
                        if j > 1
                            # Add lower-left diagonal edge:
                            # A[n, n+n_cols-1] = 1 / √2
                            push!(is, n)
                            push!(js, n + n_cols - 1)
                            push!(vs, 1 / √2)
                        end
                    end
                end
            end
        end

        A = sparse(is, js, vs, N, N)

        return A + A'         # Symmetrize
    end

    const N4 = (( 0, -1, 1.0),
                (-1,  0, 1.0),
                ( 1,  0, 1.0),
                ( 0,  1, 1.0))
    const N8 = ((-1, -1,  √2),
                ( 0, -1, 1.0),
                ( 1, -1,  √2),
                (-1,  0, 1.0),
                ( 1,  0, 1.0),
                (-1,  1,  √2),
                ( 0,  1, 1.0),
                ( 1,  1,  √2))

    """
        adjacency(R::Matrix[, neighbors::Tuple=N8]) -> SparseMatrixCSC

    Compute an adjacency matrix of the raster image `R` of the similarities/conductances
    the cells. The similarities are computes harmonic mean of the cell values weighted
    by the grid distance. The similarities can be computed with respect to eight
    neighbors (`N8`) or four neighbors (`N4`).
    """
    function adjacency(R::Matrix, neighbors::Tuple=N8)
        m, n = size(R)

        # Initialy the buffers of the SparseMatrixCSC
        is, js, vs = Int[], Int[], Float64[]

        for j in 1:n
            for i in 1:m
                # Base node
                rij = R[i, j]
                for (ki, kj, l) in neighbors
                    if !(1 <= i + ki <= m) || !(1 <= j + kj <= n)
                        # Continue when computing edge out of raster image
                        continue
                    else
                        # Target node
                        rijk = R[i + ki, j + kj]
                        if iszero(rijk)
                            # Don't include zero similaritiers
                            continue
                        end

                        push!(is, (j - 1)*m + i)
                        push!(js, (j - 1)*m + i + ki + kj*m)
                        v = 2/((inv(rij) + inv(rijk))*l)
                        push!(vs, v)
                    end
                end
            end
        end
        return sparse(is, js, vs)
    end

    function _id_to_grid_coordinate_list(N_grid, n_cols)
        id_to_grid_coordinate_list = Tuple{Int,Int}[]
        for node_id in 1:N_grid
            j = (node_id - 1) % n_cols + 1
            i = div(node_id - j, n_cols) + 1
            push!(id_to_grid_coordinate_list, (i,j))
        end
        return id_to_grid_coordinate_list
    end

    #=
    Make pixels impossible to move to by changing the affinities to them to zero.
    Input:
        - node_list: list of nodes (either node_ids or coordinate-tuples) to be made impossible
    =#
    function _set_impossible_nodes!(g::Grid, node_list::Vector{<:Tuple}, impossible_affinity=1e-20)
        # Find the indices of the coordinates in the id_to_grid_coordinate_list vector
        node_list_idx = [findfirst(isequal(n), g.id_to_grid_coordinate_list) for n in node_list]

        A = g.A

        # Set (nonzero) values to impossible_affinity:
        if impossible_affinity > 0
            A[node_list_idx,:] = impossible_affinity*(A[node_list_idx,:] .> 0)
            A[:,node_list_idx] = impossible_affinity*(A[:,node_list_idx] .> 0)
        elseif impossible_affinity == 0
            # Delete the nodes completely:
            num_of_removed = length(node_list_idx)

            nodes_to_keep = [n for n in 1:size(A, 1) if !(n in node_list_idx)]

            A = A[nodes_to_keep,:]
            A = A[:,nodes_to_keep]

            # FIXME! Commented out 8 April 2019 since qualities are now matrices.
            # Check if commention out is a problem. I don't think so.
            # deleteat!(vec(g.source_qualities), node_list_idx)
            # deleteat!(vec(g.target_qualities), node_list_idx)
            g.id_to_grid_coordinate_list = [g.id_to_grid_coordinate_list[id] for id in 1:length(g.id_to_grid_coordinate_list) if !(id in node_list_idx)]
        end

        g.A = A
    end

    """
        mapnz(f, A::SparseMatrixCSC) -> SparseMatrixCSC

    Map the non-zero values of a sparse matrix `A` with the function `f`.
    """
    function mapnz(f, A::SparseMatrixCSC)
        B = copy(A)
        map!(f, B.nzval, A.nzval)
        return B
    end

    function plot_outdegrees(g::Grid)
        values = sum(g.A, dims=2)
        canvas = zeros(g.shape...)
        for (i,v) in enumerate(values)
            canvas[g.id_to_grid_coordinate_list[i]...] = v
        end
        heatmap(canvas)
    end


    struct Habitat
        g::Grid
        C::SparseMatrixCSC{Float64,Int}
        P_ref::SparseMatrixCSC{Float64,Int}
        landmarks::Vector{Int}
    end

    """
        Habitat(g::Grid,
            costfunction::Function,
            landmarks::AbstractVector = 1:size(g.A, 1)) =
                Habitat(g,
                        mapnz(costfunction, g.A),
                        _compute_Pref(g.A),
                        landmarks)

    Construct a Habitat from a `g::Grid` based on a `costfunction`.
    """
    Habitat(g::Grid,
            costfunction::Function,
            landmarks::AbstractVector = 1:size(g.A, 1)) =
                Habitat(g,
                        mapnz(costfunction, g.A),
                        _compute_Pref(g.A),
                        landmarks)

    #=
    Compute the I - W matrix used in the free energy distance.

    Parameters:
    - β: If given separately, do not change self.I_W but only return I_W according to the given beta.
            If None, then beta=self.beta and this computes and stores self.I_W

    Returns:
    - I_W: I-W, where I is identity matrix and W is matrix with elements w_ij = P_ij*exp(-beta*c_ij)
    =#
    function _compute_W(h::Habitat; β=nothing)
        if β === nothing
            throw(ArgumentError("β must be set to a value"))
        end

        A = h.g.A
        C = h.C
        P_ref = h.P_ref
        N = LinearAlgebra.checksquare(A)

        # Compute W:
        expbC = mapnz(t -> exp(-β*t), C)
        W = P_ref .* expbC

        return W
    end

    function _compute_Pref(A::SparseMatrixCSC)
        inv_degrees = vec(sum(A, dims=1))
        map!(t -> ifelse(t > 0, inv(t), t), inv_degrees, inv_degrees)
        D_inv = spdiagm(0 => inv_degrees)
        return D_inv*A
    end

    """
        RSP_full_betweenness_qweighted(h::Habitat; β=nothing) -> Matrix

    Compute full RSP betweenness of all nodes weighted by source and target qualities.
    """
    function RSP_full_betweenness_qweighted(h::Habitat; β=nothing)
        # TODO: Verify that this works
        if β === nothing
            throw(ArgumentError("β must be set to a value"))
        else
            I_W = I - _compute_W(h, β=β)
        end

        Z = inv(Matrix(I_W))
        Zdiv = inv.(Z)
        Zdiv_diag = diag(Zdiv)

        qs = vec(h.g.source_qualities)
        qt = vec(h.g.target_qualities)
        qs_sum = sum(qs)

        ZQZdivQ = qt .* Zdiv'
        ZQZdivQ = ZQZdivQ .* qs'
        ZQZdivQ -= Diagonal(qs_sum .* qt .* Zdiv_diag)

        ZQZdivQ = Z*ZQZdivQ
        return reshape(sum(ZQZdivQ .* Z', dims=2), reverse(h.g.shape))'
    end

    """
        RSP_full_betweenness_kweighted(h::Habitat; β=nothing) -> Matrix

    Compute full RSP betweenness of all nodes weighted with proximity.
    """
    function RSP_full_betweenness_kweighted(h::Habitat; β=nothing)
        # TODO: Verify that this works
        if β === nothing
            throw(ArgumentError("β must be set to a value"))
        else
            I_W = Matrix(I - _compute_W(h, β=β))
        end

        Z = inv(I_W)

        Zdiv = inv.(Z)

        qs = vec(h.g.source_qualities)
        qt = vec(h.g.target_qualities)

        # FIXME! The python version allows for difference distance measures here. Figure out how to handle this in Julia
        K = map(t -> exp(-t), RSP_dissimilarities_to(h, β=β))
        K[diagind(K)] .= 0

        K .= qs .* K .* qt'

        K_colsum = vec(sum(K, dims=1))
        D_Zdiv = diag(Zdiv)

        ZKZdiv = K .* Zdiv
        ZKZdiv -= Diagonal(K_colsum .* D_Zdiv)

        ZKZdiv = Z*ZKZdiv
        bet = sum(ZKZdiv .* Z', dims=1)

        return reshape(bet, reverse(size(h.g)))'
    end

    """
        RSP_dissimilarities_to(h::Habitat;
                               β=nothing,
                               destinations=h.landmarks,
                               return_mean_D_KL=true,
                               algorithm=:batch) -> Matrix
    Compute RSP expected costs or RSP dissimilarities from all nodes to landmarks.
    """
    function RSP_dissimilarities_to(h::Habitat;
                                    β=nothing,
                                    destinations=h.landmarks,
                                    return_mean_D_KL=true,
                                    algorithm=:batch)
        # TODO: Not yet implemented as from_landmarks_to_all. This requires computation of the diagonal of Z, which is not trivial.
        # TODO: Do something smarter with return_mean_D_KL.
        tic = time()

        N = size(h.g.A, 1)

        N_destinations = length(destinations)

        W = _compute_W(h, β=β)

        CW = h.C .* W

        if length(destinations) == N
            # Compute all distances.
            #
            # TODO: Remove consideration of 0-quality nodes as they may cause unnecessary inf-problems
            #
            @info("Computing Z...")
            Z   = inv(Matrix(I - W))

            # TODO (maybe):
            # if self.return_mean_D_KL

            if any(iszero, Z)
                Z_has_zeros = true
                inf_idx = sparse(Z.==0)
                @warn("Z contains zeros! This will cause some distances to become infinite.")
            else
                Z_has_zeros = false
            end

            @info("Computing (C*W)Z...")
            D = Matrix(CW)
            D = D*Z

            @info("Computing Z(C*W)Z...")
            D = Z*D

            @info("Computing Z(C*W)Z/Z...")
            D ./= Z

            @info("Computing D - e*diag(D).T")
            if Z_has_zeros
                D[Matrix(inf_idx)] .= Inf
            end

            # D above actually gives the expected costs of non-hitting paths
            # from i to j.

            # Expected costs of hitting paths avoiding possible 'Inf-Inf':
            diag_vec = diag(D)
            inf_idx = isinf.(diag_vec)

            diag_vec = reshape(diag_vec, 1, length(diag_vec)) # Make into a 1xN-vector
            D .-= diag_vec

            D[:,inf_idx] .= Inf

            if return_mean_D_KL
                @info("Computing mean KL-divergence")
                Z_diag_inv = inv(Diagonal(Z))
                D_KL = Z*Z_diag_inv
                D_KL .= .-log.(D_KL) .- β.*D
                D_KL[:,inf_idx] .= 0

                mean_D_KL = (vec(h.g.source_qualities)'*D_KL)*vec(h.g.target_qualities)
            end

            toc = time() - tic
            @info("RSP expected costs computed in $toc seconds")


        elseif algorithm == :batch
            I_dest = zeros(N, N_destinations)
            I_dest[destinations, 1:N_destinations] .= 1

            Z_dest = I_W\I_dest

            unconnected_pairs = Z_dest .== 0
            if any(unconnected_pairs)
                @warn("Some values of Z_dest are zero either because of unconnected nodes or a high value of beta, which may lead to infinite distances.")
            end
            Q_dest = CW*Z_dest

            D = I_W\Q_dest

            D ./= Z_dest
            D[unconnected_pairs] .= 0
            D_dest = D[destinations, 1:N_destinations]
            D -= D_dest
            D[unconnected_pairs] .= Inf

        elseif algorithm == :sequential
            D = zeros(N, N_destinations)
            for (i,t) in enumerate(destinations)
                # I_W_t = I_W
                # I_W_t[t,:] = 0  Not really needed

                e_t = zeros(N)
                e_t[t] = 1
                z_t = I_W\e_t
                unconnected_nodes = z_t .== 0
                if any(unconnected_nodes)
                    @warn("Some values of z_t are zero either because of unconnected nodes or a high value of beta, which may lead to infinite distances.")
                end
                z_t_2d = reshape(z_t, 1, length(z_t))
                q_t = CW*z_t_2d
                c_t = I_W\q_t
                c_t ./= z_t
                c_t[unconnected_nodes] .= 0
                c_t -= c_t[t]

                D[:,i] = c_t
                D[unconnected_nodes,i] .= Inf
            end
        end

        return D
    end
end
