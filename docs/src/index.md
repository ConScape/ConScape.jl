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

Now solve the quality-weighted betweenness:

```@example 1
bet_q = ConScape.solve(Betweenness(ConScape.QualityWeighted()), g)
ConScape.heatmap(bet_q, yflip=true)
```