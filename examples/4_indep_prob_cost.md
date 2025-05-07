```julia
using ConScape
using Rasters
using ArchGDAL
using Plots
```

# Step 1: data import and Grid creation

```julia
datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
```

```julia
permeabilities = Raster(joinpath(datadir, "prob_panther_cropped.asc"); missingval=NaN)
costs = Raster(joinpath(datadir, "mort_panther_cropped.asc"); missingval=NaN)
qualities = Raster(joinpath(datadir, "prob_panther_cropped.asc"); missingval=NaN)
```

```julia
map!(t -> t < 0 ? 0 : t, costs, costs)
map!(t -> t < 0 ? 0 : t, permeabilities, permeabilities)
map!(t -> t < 0 ? 0 : t, qualities, qualities)
rast = RasterStack((; permeabilities, qualities, costs))
```

```julia
costfunction = t -> t > 0 ? -log(1 - t) : t
```

# Step 2: RSP definition

```julia
rsp = RSP(; theta=1.0)
```

show survival:

```julia
tmp = zeros(length(rast))
tmp[15000] = 1
plot(Raster(tmp, dims(rast)); title="Target (or is it Source?) Pixel")
```

```julia
sp = solve(SurvivalProbability(), rsp, rast)
plot(sp)
```

```julia
survival_vec = map(t -> t == 1 ? NaN : t, sp[:, 15000])
plot(Raster(survival_vec, dims(rast)); title="Survival Probability")
```

```julia
rsp = RandomisedShortestPath(; 
    proximity_measure=SurvivalProbability(), 
    distance_transformation=ConScape.ExpMinus(),
    theta=1.0
)
ch = solve(ConnectedHabitat(), rsp, rast)
```

```julia
plot(ch; title="Cumulative Connected Habitat (Survival)")
```