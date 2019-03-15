module ConScape

    using SparseArrays, PyPlot

    mutable struct Grid
        shape::Tuple{Int,Int}
        A::SparseMatrixCSC{Float64,Int}
        id_to_grid_coordinate_list::Vector{Tuple{Int,Int}}
    end

    function Grid(;shape=nothing, qualities=nothing, nhood_size=8)
        N_grid = prod(shape)
        n_cols = shape[2]
        Grid(shape,
             _generateA(shape..., nhood_size),
             _set_id_to_grid_coordinate_list(N_grid, n_cols))
    end

    """
    simulate a permeable wall
    """
    function perm_wall_sim(;grid_shape=(30,60), Q=1, A=0.5, ww=3, wp=0.5, cw=(3,3), cp=(0.35,0.7))

        # 1. initialize landscape
        n_rows, n_cols = grid_shape
        N = n_rows*n_cols
        qualities = ones(N)*Q
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

    function plot_outdegrees(g::Grid; cmap="cool")
        values = sum(g.A, dims=2)
        canvas = zeros(g.shape...)
        for (i,v) in enumerate(values)
            canvas[g.id_to_grid_coordinate_list[i]...] = v
        end
        imshow(canvas, cmap=cmap)
    end
end
