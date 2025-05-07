
```julia
using ConScape
using Plots
using Rasters
using ArchGDAL
```

# Step 1: data import and RasterStack creation

First source the example data from the package:

```julia
datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
```

We need to specify the correct layer names `source_qualities` 
and `affinities` so that ConScape.jl can find them.

To do this we make a `NamedTuple` of the file paths:

```julia
paths = (; qualities=joinpath(datadir, "hab_qual_1000.asc"), 
           permeabilities=joinpath(datadir, "mov_prob_1000.asc"))
rast = RasterStack(paths; missingval=NaN)
```

```julia
# Set values close to very where they are missing :
matches = boolmask(rast)
rast.qualities[Not(matches)] .= 1e-20
rast.permeabilities[Not(matches)] .= 1e-20;
```

We can plot the whole raster:

```julia
plot(rast)
```

# Step 2: Define the problem

Here we define a problem with randomised shortest path movement,
with a theta value of `1.0`, and otherwise use the defaults.

```julia
rsp = RandomisedShortestPath(; theta=1.0)
```

# Solve a proximity measure: `ExpectedCost()`

`ProximityMeasures` return outputs in graph-space, so that a 
`(n * m)` raster input gives an `(n * m, n * m)` sparse output.

Now we solve as a single measure. This isn't the most efficient 
way to work in production, but is easy for exploration.

There are many ways to call `solve`, but generally the method has the 
measure first, movement mode or problem in the middle, and spatial data last.

```julia
dists = solve(ExpectedCost(), rsp, rast)
```

```julia
plot(Raster(dists[:, 4300], dims(rast)), title="RSP Expected Cost Distance")
```

```julia
plot(Raster(map(x -> exp(-x/75), dists[:, 4300]), dims(rast)), title="Proximity")
```

# Step 3 Solve a spatial measure: Amount of Connected Habitat

Individual `SpatialMeasure`s return a spatial `Raster` with 
the same dimensions as the input `RasterStack`.

We now use the shorthand `RSP` for `RandomisedShortestPath`,
and define it with a custom `distance_transformation`:

```julia
rsp = RSP(; theta=1.0, distance_transformation=ExpMinusAlpha(75));
ch = solve(ConnectedHabitat(), rsp, rast)
```

```julia
plot(ch; title="Connected Habitat")
```

```julia
sum(ch)
```

# Step 4: Movement Flow

```julia
bq = solve(Betweenness(QualityWeighted()), rsp, rast)
plot(bq; title="Quality-weighted Movement Flow")
```

Effect of theta:

```julia
# Make an empty raster
qualities = zeros(dims(rast))
# Assign two pixels with non-zero quality
qualities[58, 42] = 1
qualities[90, 66] = 1
# qualities[70, 60] = 1
# qualities[105, 50] = 1

# Build a RasterStack input using the original permeabilities
test_rast = RasterStack((; qualities, permeabilities=rast.permeabilities))
```

Now calculate betweeness for multiple `theta` values:

```julia
thetas = (2.5, 1.0, 0.5, 0.1, 0.01, 0.001)
betq_layers = map(thetas) do theta
    solve(Betweenness(QualityWeighted()), RSP(; theta), test_rast) 
end
```

And plot them, for comparison:

```julia
betq = RasterStack(betw_layers; name=map(θ -> "theta_$θ", thetas))
plot(betq; size=(800, 400))
# savefig("output_figures/figure_thetas.png")
```

## Quality and proximity weighted

```julia
rsp = RSP(; theta=1.0, distance_transformation=ExpMinusAlpha(75))
betm = solve(Betweenness(QualityAndProximityWeighted()), rsp, rast)
plot(betm, title="Quality and Proximity Weighted Movement Flow") 
```

Again calculate betweeness for multiple `theta` values:

```julia
betm_layers = map(thetas) do theta
    solve(Betweenness(QualityAndProximityWeighted()), RSP(; theta), test_rast) 
end
```

And plot them, for comparison:

```julia
betm = RasterStack(betm_layers; name=map(θ -> "theta_$θ", thetas))
plot(betm; size=(800, 400))
# savefig("output_figures/figure_thetas.png")
```