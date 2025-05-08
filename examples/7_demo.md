---
title: ConScape demo
author: Bram Van Moorter and Andreas Noack
---

TODO: delete this file


This is a small demo of the functionalty in the `ConScape` package for Julia.

First we need to load the package

```julia
using ConScape, Plots, Graphs
```

```julia
g = ConScape.permeable_wall_sim(30, 60; 
    corridorwidths=(3, 2),
    qualities=collect(reshape(1800:-1:1, 60, 30)'),
)
```

```julia
heatmap(g.source_quality_spatial)
```

```julia
rsp = RSP(; theta=0.2)
```

```julia
cost_graph = ConScape.SimpleWeightedGraph(g.costmatrix)
@time distances_all2L_shortestpath = Graphs.floyd_warshall_shortest_paths(cost_graph).dists
```

```julia
map!(distances_all2L_shortestpath, distances_all2L_shortestpath) do t
    (isfinite(t) && !iszero(t)) ? exp(-t) : t 
end
```

```julia
@time bet_shortest = Graphs.betweenness_centrality(Cg)
heatmap(reshape(bet_shortest, 30, 60), yflip=true)
```

```julia
betq = solve(Betweenness(QualityWeighted()), rsp, raster)
heatmap(betq; yflip=true)
```

```julia
ec = solve(ExpectedCost(), rsp, raster)
```

```julia
betm = solve(Betweenness(QualityAndProximityWeighted()), rsp, g);
plot(betm)
```