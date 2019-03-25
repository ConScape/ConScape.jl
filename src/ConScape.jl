module ConScape

    using SparseArrays, PyPlot, LightGraphs, SimpleWeightedGraphs
    using LinearAlgebra

    mutable struct Grid
        shape::Tuple{Int,Int}
        A::SparseMatrixCSC{Float64,Int}
        id_to_grid_coordinate_list::Vector{Tuple{Int,Int}}
        qualities::Vector{Float64}
    end

    function Grid(;shape=nothing, qualities::Vector=ones(prod(shape)), nhood_size=8)
        N_grid = prod(shape)
        n_cols = shape[2]
        Grid(shape,
             _generateA(shape..., nhood_size),
             _set_id_to_grid_coordinate_list(N_grid, n_cols),
             qualities)
    end

    """
    simulate a permeable wall
    """
    function perm_wall_sim(;grid_shape=(30,60), Q=1, A=0.5, ww=3, wp=0.5, cw=(3,3), cp=(0.35,0.7), qualities=fill(Q, prod(grid_shape)))

        # 1. initialize landscape
        n_rows, n_cols = grid_shape
        N = n_rows*n_cols
        g = Grid(shape=grid_shape, qualities=qualities)
        g.A = A * g.A

        # # 2. compute the wall
        wpt = floor(Int, n_cols*wp) - ceil(Int, ww/2)
        xs = range(wpt, stop=wpt+ww)

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

    """
    Generate the affinity matrix of a grid graph, where each
    pixel is connected to its vertical and horizontal neighbors.

    Parameters:
    - nhood_size: 4 creates horizontal and vertical edges, 8 creates also diagonal edges
    """
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

    function _set_id_to_grid_coordinate_list(N_grid, n_cols)
        id_to_grid_coordinate_list = []
        for node_id in 1:N_grid
            j = (node_id - 1) % n_cols + 1
            i = div(node_id - j, n_cols) + 1
            push!(id_to_grid_coordinate_list, (i,j))
        end
        return id_to_grid_coordinate_list
    end

    """
    Make pixels impossible to move to by changing the affinities to them to zero.
    Input:
        - node_list: list of nodes (either node_ids or coordinate-tuples) to be made impossible
    """
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

            # MORE COMPLICATED VERSION OF ABOVE:
            ##################
            # sh = A.shape

            # # Delete rows:
            # A.rows = np.delete(A.rows, node_list_idx)
            # A.data = np.delete(A.data, node_list_idx)
            # A._shape = (sh[0]-num_of_removed, sh[1])

            # # Delete columns:
            # A = A.T
            # sh = A.shape
            # A.rows = np.delete(A.rows, node_list_idx)
            # A.data = np.delete(A.data, node_list_idx)
            # A._shape = (sh[0]-num_of_removed, sh[1])
            # A = A.T
            ###################

            deleteat!(g.qualities, node_list_idx)
            g.id_to_grid_coordinate_list = [g.id_to_grid_coordinate_list[id] for id in 1:length(g.id_to_grid_coordinate_list) if !(id in node_list_idx)]
        end

        g.A = A
    end

    function mapnz(f, A::SparseMatrixCSC)
        B = copy(A)
        map!(f, B.nzval, A.nzval)
        return B
    end

    function plot_outdegrees(g::Grid; cmap="cool")
        values = sum(g.A, dims=2)
        canvas = zeros(g.shape...)
        for (i,v) in enumerate(values)
            canvas[g.id_to_grid_coordinate_list[i]...] = v
        end
        imshow(canvas, cmap=cmap)
    end


    struct HabitatAnalysis
        g::Grid
        C::SparseMatrixCSC{Float64,Int}
        P_ref::SparseMatrixCSC{Float64,Int}
        landmarks::Vector{Int}
    end

    HabitatAnalysis(g::Grid,
                    costfunction::Function,
                    landmarks::AbstractVector = 1:size(g.A, 1)) =
                        HabitatAnalysis(g,
                                        mapnz(costfunction, g.A),
                                        _compute_Pref(g.A),
                                        landmarks)

    ```
    Compute the I - W matrix used in the free energy distance.

    Parameters:
    - β: If given separately, do not change self.I_W but only return I_W according to the given beta.
            If None, then beta=self.beta and this computes and stores self.I_W

    Returns:
    - I_W: I-W, where I is identity matrix and W is matrix with elements w_ij = P_ij*exp(-beta*c_ij)
    ```
    function _compute_W(h::HabitatAnalysis; β=nothing)
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

    ```
    Compute full RSP betweenness of all nodes weighted by quality
    TODO: Verify that this works
    ```
    function RSP_full_betweenness_qweighted(h::HabitatAnalysis; β=nothing)
        if β === nothing
            throw(ArgumentError("β must be set to a value"))
        else
            I_W = I - _compute_W(h, β=β)
        end

        Z = inv(Matrix(I_W))
        # tol = 1/1000000 #used to avoid 0 division
        # Zdiv = map(t -> ifelse(t > tol, inv(t), inv(tol)), Z)
        Zdiv = inv.(Z)

        # Qs = Diagonal(h.g.qualities)
        # Qt = Qs
        # sm = sum(h.g.qualities)

        # D_Zdiv = Diagonal(diag(Zdiv * Qt))

        # bet = diag( Z * (Qt * Zdiv' * Qs - sm * D_Zdiv) * Z )

        q = h.g.qualities
        sm = sum(q)
        D_Zdiv = Diagonal(diag(Zdiv) .* q)
        bet = diag(Z * (q .* Zdiv' .* q' .- sm .* D_Zdiv) * Z)

        return bet
    end

    ```
    Compute full RSP betweenness of all nodes weighted with proximity
    TODO: Verify that this works
    ```
    function RSP_full_betweenness_kweighted(h::HabitatAnalysis, β=nothing)
        if β === nothing
            throw(ArgumentError("β must be set to a value"))
        else
            I_W = Matrix(I - _compute_W(h, β=β))
        end

        Z = inv(I_W)

        # if Z.min() == 0:
        #     Z[Z < 1./1000000] = 1./1000000 #used to avoid 0 division
        Zdiv = inv.(Z)

        q = h.g.qualities

        K = _similarities_all2L(h)

        K = Qs*(K*Qt)
        e = ones(length(q), length(q))

        # TODO: Check that this is written correctly, especially concerning the elementwise and dot products:
        bet = diag( Z * ((Zdiv*K)' .- diag(Zdiv) * diag(K' * e)) * Z )

        return bet
    end

    ```
    Compute RSP expected costs or RSP dissimilarities from all nodes to landmarks.

    TODO: Not yet implemented as from_landmarks_to_all. This requires computation of the diagonal of Z, which is not trivial.
    TODO: Do something smarter with return_mean_D_KL.
    ```
    function RSP_dissimilarities_to(h::HabitatAnalysis; β=nothing, destinations=h.landmarks, return_mean_D_KL=true, algorithm=:batch)
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
                @warn("Computing mean KL-divergence")
                Z_diag_inv = inv(Diagonal(Z))
                D_KL = Z*Z_diag_inv
                D_KL .= .-log.(D_KL) .- β.*D
                D_KL[:,inf_idx] .= 0

                mean_D_KL = (h.g.qualities'*D_KL)*h.g.qualities
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
