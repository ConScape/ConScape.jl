# ConScape.jl

A Julia package for landscape connectivity.

## Example

In the following, a small example of the functionality is given. First, we canstruct an artificial landscape with a wall and two corridors. The landscape is stored in a `Grid` struct.

```@example 1
using ConScape
g = ConScape.permeable_wall_sim(30, 60; 
    corridorwidths=(3,2), costfunction=ConScape.MinusLog()
); # Generate an artificial landscape
heatmap(ConScape.outdegrees(g))
```

From a `Grid`, we can now create a `GridRSP` which we can use to compute the randomized shortest path based quality weighted betweenness with the temperature parameter `θ=0.2`.

```@example 1
bet_q = ConScape.solve(Betweenness(ConScape.QualityWeighted()), g)
ConScape.heatmap(bet_q, yflip=true)
```