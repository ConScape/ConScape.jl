# simulate a permeable wall
function perm_wall_sim(
    nrows::Integer,
    ncols::Integer;
    scaling::Float64=0.5,
    wallwidth::Integer=3,
    wallposition::Float64=0.5,
    corridorwidths::NTuple{<:Any,Int}=(3,3),
    corridorpositions=(0.35,0.7),
    impossible_affinity::Real=1e-20,
    nhood_size::Integer=8,
    kwargs...)

    # 1. initialize landscape
    affinities = _generate_affinities(nrows, ncols, nhood_size)
    g = Grid(nrows, ncols; affinities=affinities*scaling, kwargs...)

    # # 2. compute the wall
    wpt = round(Int, ncols*wallposition - wallwidth/2 + 1)
    xs  = range(wpt, stop=wpt + wallwidth - 1)

    # 3. compute the corridors
    ys = Int[]
    for i in 1:length(corridorwidths)
        cpt = floor(Int, nrows*corridorpositions[i]) - ceil(Int, corridorwidths[i]/2)
        if i == 1
            append!(ys, 1:cpt)
        else
            append!(ys, range(maximum(ys) + 1 + corridorwidths[i-1], stop=cpt))
        end
    end
    append!(ys, range(maximum(ys) + 1 + corridorwidths[end]  , stop=nrows))

    impossible_nodes = vec(CartesianIndex.(collect(Iterators.product(ys, xs))))
    g = _set_impossible_nodes(g, impossible_nodes, impossible_affinity)

    return g
end

⊗ = kron
#=
Generate the affinity matrix of a grid graph, where each
pixel is connected to its vertical and horizontal neighbors.

Parameters:
- nhood_size: 4 creates horizontal and vertical edges, 8 creates also diagonal edges
=#
function _generate_affinities(nrows, ncols, nhood_size)

    if !(nhood_size ∈ (4, 8))
        throw(ArgumentError("nhood_size must be either 4 or 8"))
    end

    affinities = spdiagm(0=>ones(ncols)) ⊗ spdiagm(-1=>ones(nrows - 1), 1=>ones(nrows - 1)) +
        spdiagm(-1=>ones(ncols - 1), 1=>ones(ncols - 1)) ⊗ spdiagm(0=>ones(nrows))

    if nhood_size == 8
        affinities .+= spdiagm(-1=>ones(ncols - 1), 1=>ones(ncols - 1)) ⊗
              spdiagm(-1=>fill(1/√2, nrows - 1), 1=>fill(1/√2, nrows - 1))
    end

    return affinities
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

@enum AdjacencyWeight begin
    TargetWeight
    AverageWeight
end

@enum MatrixType begin
    AffinityMatrix
    CostMatrix
end

"""
    graph_matrix_from_raster(R::Matrix[, type=AffinityMatrix, neighbors::Tuple=N8, weight=TargetWeight])::SparseMatrixCSC

Compute a graph matrix, i.e. an affinity or cost matrix of the raster image `R` of cell affinities or cell costs.
The values are computed as either the value of the target cell (TargetWeight) or as harmonic (arithmetic) means
of the cell affinities (costs) weighted by the grid distance (AverageWeight). The values can be computed with
respect to eight neighbors (`N8`) or four neighbors (`N4`).
"""
function graph_matrix_from_raster(
    R::Matrix;
    matrix_type=AffinityMatrix,
    neighbors::Tuple=N8,
    weight=TargetWeight
)
    m, n = size(R)

    # Initialize the buffers of the SparseMatrixCSC
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
                    if iszero(rijk) || isnan(rijk)
                        # Don't include zero or NaN similaritiers
                        continue
                    end

                    push!(is, (j - 1)*m + i)
                    push!(js, (j - 1)*m + i + ki + kj*m)
                    if weight == TargetWeight
                        if matrix_type == AffinityMatrix
                            push!(vs, rijk/l)
                        elseif matrix_type == CostMatrix
                            push!(vs, rijk*l)
                        end
                    elseif weight == AverageWeight
                        if matrix_type == AffinityMatrix
                            v = 2/((inv(rij) + inv(rijk))*l)
                            push!(vs, v)
                        elseif matrix_type == CostMatrix
                            v = ((rij + rijk)*l)/2
                            push!(vs, v)
                        end
                    else
                        throw(ArgumentError("weight mode $weight not implemented"))
                    end
                end
            end
        end
    end
    return sparse(is, js, vs, m*n, m*n)
end


#=
Make pixels impossible to move to by changing the affinities to them to zero.
Input:
    - node_list: list of nodes (either node_ids or coordinate-tuples) to be made impossible
=#
function _set_impossible_nodes(g::Grid, node_list::Vector{CartesianIndex{2}}, impossible_affinity=1e-20)
    # Find the indices of the coordinates in the id_to_grid_coordinate_list vector
    node_list_idx = [findfirst(isequal(n), g.id_to_grid_coordinate_list) for n in node_list]

    # Copy affinities and qualities for modification
    affinities       = copy(g.affinities)
    source_qualities = copy(g.source_qualities)
    target_qualities = copy(g.target_qualities)

    # Set (nonzero) values to impossible_affinity:
    # Affinities
    # FIXME! Row slicing of a sparse matrix is really inefficient
    affinities[node_list_idx,:] = impossible_affinity*(affinities[node_list_idx,:] .> 0)
    affinities[:,node_list_idx] = impossible_affinity*(affinities[:,node_list_idx] .> 0)
    dropzeros!(affinities)

    # Qualities
    source_qualities[node_list] .= 0
    target_qualities[node_list] .= 0

    # Generate a new Grid based on the modified affinities
    return Grid(size(g)...,
        affinities=affinities,
        source_qualities=source_qualities,
        target_qualities=target_qualities,
        costs=g.costfunction === nothing ? g.costmatrix : g.costfunction)
end

"""
    mapnz(f, A::SparseMatrixCSC)::SparseMatrixCSC

Map the non-zero values of a sparse matrix `A` with the function `f`.
"""
function mapnz(f, A::SparseMatrixCSC)
    B = copy(A)
    map!(f, B.nzval, A.nzval)
    return B
end

"""
    make_landscape(pth::String; q::String, a::String, c::Union{String,Nothing})::Array

Read asc files of raster data from a directory, into a 3D Array of quality, affinity, and optionally cost.
The moving_window function takes the output from this function.
"""

function make_landscape(pth::String; q::String, a::String, c::Union{String,Nothing}=nothing)
    mov_prob, meta_p = readasc(joinpath(pth, a))
    hab_qual, meta_q = readasc(joinpath(pth, q))
    
    if collect(values(meta_p))[1:end .!= 3] != collect(values(meta_q))[1:end .!= 3]
        throw(ArgumentError("Maps of quality and permeability do not match"))
    end
    
    if c === nothing
        lnd = reshape([hab_qual mov_prob], size(hab_qual)[1], size(hab_qual)[2], :)
    else
        mov_cost, meta_c = readasc(joinpath(pth, c))
        if collect(values(meta_p))[1:end .!= 3] != collect(values(meta_c))[1:end .!= 3]
            throw(ArgumentError("The cost map does not match with the other maps"))
        end
        lnd = reshape([hab_qual mov_prob mov_cost], size(hab_qual)[1], size(hab_qual)[2], :)
    end    
    
    return lnd
end

"""
    moving_window_helper(win::Tuple{Int64, Int64}; dwin::Dict, win_cntr::Matrix, lnd::Array, cst_fun::Union{Transformation,Nothing}, fnc::String, args::Dict)::SparseMatrixCSC
	
The serial version of the helper function. The function clips the window from the landscape and computes the desired function. The result is returned as a SparseMatrix, which are combined in the main moving_window function.  
"""

function moving_window_helper(win::Tuple{Int64, Int64}; dwin::Dict, win_cntr::Matrix, lnd::Array, cst_fun::Union{Transformation,Nothing}, fnc::String, args::Dict)
    res = zeros(size(lnd)[1:2])

    tgt_lms = [[((win[1]-1)*dwin["d"])+1, minimum([(win[1]*dwin["d"]), size(lnd)[1]])], 
        [((win[2]-1)*dwin["d"])+1, minimum([(win[2]*dwin["d"]), size(lnd)[2]])]]
    src_lms = [[maximum([tgt_lms[1][1]-dwin["buf"],1]), minimum([tgt_lms[1][2]+dwin["buf"], size(lnd)[1]])], 
        [maximum([tgt_lms[2][1]-dwin["buf"],1]), minimum([tgt_lms[2][2]+dwin["buf"], size(lnd)[2]])]]
    
    lnd_clp = lnd[src_lms[1][1]:src_lms[1][2],src_lms[2][1]:src_lms[2][2],:]
    tgts = zeros(size(lnd_clp[:,:,1]))
    tgts[(dwin["buf"]+1):minimum([dwin["buf"]+dwin["d"], size(tgts)[1]]),
                        (dwin["buf"]+1):minimum([dwin["buf"]+dwin["d"], size(tgts)[2]])] = 
                    map(t -> isnan(t) ? 0.0 : t, lnd[tgt_lms[1][1]:tgt_lms[1][2],tgt_lms[2][1]:tgt_lms[2][2],1]) .* 
                                        win_cntr[1:(tgt_lms[1][2]-tgt_lms[1][1]+1),1:(tgt_lms[2][2]-tgt_lms[2][1]+1)]                                                            
    tgts = dropzeros(sparse(tgts));
    
    #skip empty affinities clips
    if !((sum(map!(x -> isnan(x) ? 0 : x, lnd_clp[:,:,1], lnd_clp[:,:,1])) == 0) || 
        (sum(map!(x -> isnan(x) ? 0 : x, lnd_clp[:,:,2], lnd_clp[:,:,2])) == 0) || 
        length(nonzeros(tgts)) ==0)
        
        adjacency_matrix = graph_matrix_from_raster(lnd_clp[:,:,2]);
        # compute costs, if missing
        if (size(lnd_clp)[3]==2)    
            cost_matrix = mapnz(cst_fun, adjacency_matrix)
        else 
            cost_matrix = graph_matrix_from_raster(lnd_clp[:,:,3]);
        end
        
        
        h = GridRSP(Grid(size(lnd_clp[:,:,2])...,
                    affinities=adjacency_matrix,
                    source_qualities=lnd_clp[:,:,1],
                    target_qualities=tgts,
                    costs=cost_matrix, prune=true, verbose=false), θ=get(args, "θ", 1))
        
        if fnc === "connected_habitat"
            tmp = connected_habitat(h, 
                connectivity_function=get(args, "connectivity_function", expected_cost), 
                distance_transformation=get(args, "connectivity_function", ExpMinus()));
        elseif fnc === "betweenness_kweighted"
            tmp = betweenness_kweighted(h, 
                connectivity_function=get(args, "connectivity_function", expected_cost), 
                distance_transformation=get(args, "connectivity_function", ExpMinus()));
        elseif fnc === "betweenness_qweighted"
            tmp = betweenness_qweighted(h);
        else 
            throw(ArgumentError("Currently only the connected_habitat and both betweenness functions are supported"))
        end
     
        # return the result as a sparce matrix
        res[src_lms[1][1]:src_lms[1][2],src_lms[2][1]:src_lms[2][2]] = map(t -> isnan(t) ? 0.0 : t, tmp)
    end
    res = sparse(res)
    return(res)
end

"""
    clip_landscape(win::Tuple{Int64, Int64}; dwin::Dict, win_cntr::Matrix, lnd::Array)::Dict

Data preparation function for the parallel version of the moving_window_helper. It clips a window (win) from the landscape (lnd), which is returned as a Dictionary.
The parameters for the window computation are provided in the window dictionary (dwin).
The dictionary with the clipped landscape contains the boundaries or limits (i.e. upper and lower rows and columns) for the targets (tgt_lms) and sources (src_lms) together with the clipped landscape array (lnd_clp) and a sparce targets matrix (tgts). The targets matrix is the same size as the clipped landscape, but has only non-zero values on the non-zero values in the window center. 
"""

function clip_landscape(win::Tuple{Int64, Int64}; dwin::Dict, win_cntr::Matrix, lnd::Array)
    tgt_lms = [[((win[1]-1)*dwin["d"])+1, minimum([(win[1]*dwin["d"]), size(lnd)[1]])], 
        [((win[2]-1)*dwin["d"])+1, minimum([(win[2]*dwin["d"]), size(lnd)[2]])]]
    src_lms = [[maximum([tgt_lms[1][1]-dwin["buf"],1]), minimum([tgt_lms[1][2]+dwin["buf"], size(lnd)[1]])], 
        [maximum([tgt_lms[2][1]-dwin["buf"],1]), minimum([tgt_lms[2][2]+dwin["buf"], size(lnd)[2]])]]
    
    lnd_clp = lnd[src_lms[1][1]:src_lms[1][2],src_lms[2][1]:src_lms[2][2],:]
    tgts = zeros(size(lnd_clp[:,:,1]))
    tgts[(dwin["buf"]+1):minimum([dwin["buf"]+dwin["d"], size(tgts)[1]]),
                        (dwin["buf"]+1):minimum([dwin["buf"]+dwin["d"], size(tgts)[2]])] = 
                    map(t -> isnan(t) ? 0.0 : t, lnd[tgt_lms[1][1]:tgt_lms[1][2],tgt_lms[2][1]:tgt_lms[2][2],1]) .* 
                                        win_cntr[1:(tgt_lms[1][2]-tgt_lms[1][1]+1),1:(tgt_lms[2][2]-tgt_lms[2][1]+1)]                                                            
    tgts = dropzeros(sparse(tgts));
    res = Dict("tgt_lms" => tgt_lms, "src_lms" => src_lms, "clp" => lnd_clp, "tgts" => tgts)
    return(res)
end

"""
    moving_window_helper(lnd_clp::Dict; cst_fun::Union{Transformation,Nothing}, fnc::String, args::Dict, lnd_sz::Tuple{Int64, Int64})::SparseMatrixCSC

Helper function for the parallel version of the moving_window, it takes the dictionary of the clipped landscape (lnd_clp) as the main input (this dictionary is the output from the clip_landscape function). 
The size of the original landscape is provided as a Tupple (lnd_sz) to return the results to the main function. 
"""
function moving_window_helper(lnd_clp::Dict; cst_fun::Union{Transformation,Nothing}, fnc::String, args::Dict, lnd_sz::Tuple{Int64, Int64})
    res = zeros(lnd_sz)
    
    #skip empty affinities clips
    if !((sum(map!(x -> isnan(x) ? 0 : x, lnd_clp["clp"][:,:,1], lnd_clp["clp"][:,:,1])) == 0) || 
        (sum(map!(x -> isnan(x) ? 0 : x, lnd_clp["clp"][:,:,2], lnd_clp["clp"][:,:,2])) == 0) || 
        length(nonzeros(lnd_clp["tgts"])) ==0)
        
        adjacency_matrix = graph_matrix_from_raster(lnd_clp["clp"][:,:,2]);
        # compute costs, if missing
        if (size(lnd_clp["clp"])[3]==2)    
            cost_matrix = mapnz(cst_fun, adjacency_matrix)
        else 
            cost_matrix = graph_matrix_from_raster(lnd_clp["clp"][:,:,3]);
        end
        
        
        h = GridRSP(Grid(size(lnd_clp["clp"][:,:,2])...,
                    affinities=adjacency_matrix,
                    source_qualities=lnd_clp["clp"][:,:,1],
                    target_qualities=lnd_clp["tgts"],
                    costs=cost_matrix, prune=true, verbose=false), θ=get(args, "θ", 1))
        
        if fnc === "connected_habitat"
            tmp = connected_habitat(h, 
                connectivity_function=get(args, "connectivity_function", expected_cost), 
                distance_transformation=get(args, "connectivity_function", ExpMinus()));
        elseif fnc === "betweenness_kweighted"
            tmp = betweenness_kweighted(h, 
                connectivity_function=get(args, "connectivity_function", expected_cost), 
                distance_transformation=get(args, "connectivity_function", ExpMinus()));
        elseif fnc === "betweenness_qweighted"
            tmp = betweenness_qweighted(h);
        else 
            throw(ArgumentError("Currently only the connected_habitat and both betweenness functions are supported"))
        end
     
        # return the result as a sparce matrix
        res[lnd_clp["src_lms"][1][1]:lnd_clp["src_lms"][1][2],lnd_clp["src_lms"][2][1]:lnd_clp["src_lms"][2][2]] = 
                                                                                        map(t -> isnan(t) ? 0.0 : t, tmp)
    end
    res = sparse(res)
    return(res)
end

"""
    moving_window(win_buf::Integer; win_cntr:::Matrix, lnd::Array, cst_fun::Union{Transformation,Nothing}, fnc::String, args::Dict, parallel::bool)::Array

moving_window function to run connected_habitat or betweenness functions (fnc) as a moving window over the landscape (lnd, an output from make_landscape). 
It relies on three helper functions:
1) make_landscape: which creates an Array with a quality, affinity, and optionally a cost layer
2) clip_landscape: which clips a window from the landscape and returns it as a dictionary
3) moving_window_helper: which runs the actual computations for each window.

The main moving_window function creates a window dictionary (dwin) with the window buffer (buf), the distance (d) between windows, a tupple (n) with the number of windows row- and columnwise, and the size (sz) of the window.
Windows are identified as a tupple for the row and column direction. It dispatches the computation to the moving_window_helper based on whether computation should be parallelized or not. If parallel then the landscape will first be clipped to reduce memory useage and these clips will be processed in parallel. If serial (parallel=false) then the clipping will occur within the helper function. Thus, dispatch occurs based on whether the main argument is a dictionary with landscape clips or a tupple with the window identification. The helpers return a sparse array, which is then squashed along the window axis to return the final results of the computation.

The moving window requires a center (win_cntr), which is a square matrix of zeros and ones, denoting the targets. The size of the window is given by the size of the win_cntr and the padding or buffer around the center: win_buf. The arguments for the fnc are to be provided in args. Finally, the function can be ran in parallel.
"""

function moving_window(win_buf::Integer=5; win_cntr::Matrix=[1], lnd::Array, cst_fun::Union{Transformation,Nothing}=nothing, fnc::String, args::Dict, parallel::Bool=false)
    if length(size(win_cntr)) === 2
        if size(win_cntr)[1] != size(win_cntr)[2]
            throw(ArgumentError("Only square windows and window centers are currently supported"))
        end
    end
    
    dwin = Dict("buf" => win_buf, "d" => size(win_cntr)[1], "n" => Int.(ceil.(size(lnd)[1:2] ./ size(win_cntr)[1])))
    dwin["sz"] = (dwin["buf"]*2+dwin["d"])
    wins = vec(collect(Iterators.product(1:dwin["n"][1], 1:dwin["n"][2])))
    
    if !(fnc in ["connected_habitat", "betweenness_qweighted", "betweenness_kweighted"])
        throw(ArgumentError("Currently only the connected_habitat and betweenness functions are supported"))
    end
    
    if parallel
        lnd_clps = [clip_landscape(win; dwin=dwin, win_cntr=win_cntr, lnd=lnd) for win in wins]
        res = ThreadsX.map(lnd_clp -> moving_window_helper(lnd_clp; cst_fun=cst_fun, fnc=fnc, args=args, lnd_sz=size(lnd)[1:2]), lnd_clps)
    else
        #lnd_clps = [clip_landscape(win; dwin=dwin, win_cntr=win_cntr, lnd=lnd) for win in wins]
        #res = [moving_window_helper(lnd_clp; cst_fun=cst_fun, fnc=fnc, args=args) for lnd_clp in lnd_clps]
        res = [moving_window_helper(win; dwin=dwin, win_cntr=win_cntr, lnd=lnd, cst_fun=cst_fun, fnc=fnc, args=args) for win in wins]
    end
    
    res = reshape(vcat(res...), size(lnd)[1], :, size(lnd)[2])
    res = sum(res, dims=2)[:,1,:];
    return(res)
end
