# ConScape.jl

A Julia package for landscape connectivity.

The current version of this package is derived from the Python package [reindeers](https://bitbucket.org/rdevooght/reindeers.git).

## Example

In the following, a small example of the functionality is given. First, we canstruct an artificial landscape with a wall and two corridors. The landscape is stored in a `Grid` struct.

```@example 1
using ConScape
g = ConScape.perm_wall_sim(30, 60, corridorwidths=(3,2)) # Generate an artificial landscape
ConScape.plot_outdegrees(g)
```

From a `Grid` and a `costfunction`, we can now create a `Habitat` which we can use to compute the randomized shortest path based quality weighted betweenness with the temperature parameter `β=0.2`.

```@example 1
h = ConScape.Habitat(g, ConScape.MinusLog());
bet_q = ConScape.RSP_full_betweenness_qweighted(h, β=0.2)
ConScape.heatmap(bet_q, yflip=true)
```

## Details

The section provides more details about the functions of this package including exact call signatures.

### Grid
```@docs
ConScape.Grid
ConScape.adjacency
```

### Habitat
```@docs
ConScape.Habitat
ConScape.RSP_dissimilarities
ConScape.RSP_full_betweenness_qweighted
ConScape.RSP_full_betweenness_kweighted
ConScape.mean_kl_distance
```

### Utility functions
```@docs
ConScape.mapnz
ConScape.readasc
```
