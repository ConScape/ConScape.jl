# ConScape.jl

A Julia package for landscape connectivity.

## Example

In the following, a small example of the functionality is given. First, we canstruct an artificial landscape with a wall and two corridors. The landscape is stored in a `Grid` struct.

```@example 1
using ConScape
g = ConScape.perm_wall_sim(30, 60, corridorwidths=(3,2)) # Generate an artificial landscape
ConScape.plot_outdegrees(g)
```

From a `Grid` and a `costfunction`, we can now create a `GridRSP` which we can use to compute the randomized shortest path based quality weighted betweenness with the temperature parameter `β=0.2`.

```@example 1
h = ConScape.GridRSP(g, cost=ConScape.MinusLog(), β=0.2)
bet_q = ConScape.betweenness_qweighted(h)
ConScape.heatmap(bet_q, yflip=true)
```

## Details

The section provides more details about the functions of this package including exact call signatures.

```@meta
DocTestSetup = quote
    using ConScape
end
```

### Grid
```@docs
ConScape.Grid
ConScape.is_connected
ConScape.largest_subgraph
ConScape.least_cost_distance
```

### Habitat
```@docs
ConScape.GridRSP
ConScape.dissimilarities
ConScape.betweenness_qweighted
ConScape.betweenness_kweighted
ConScape.mean_kl_divergence
ConScape.mean_lc_kl_divergence
ConScape.least_cost_kl_divergence
ConScape.functionality
ConScape.criticality
ConScape.LF_sensitivity
```

### Utility functions
```@docs
ConScape.graph_matrix_from_raster
ConScape.mapnz
ConScape.readasc
```

This package is derived from the Python package [reindeers](https://bitbucket.org/rdevooght/reindeers.git).
