
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
permabilities = Raster(joinpath(datadir, "mov_prob_1000.asc"); missingval=NaN)
qualities = Raster(joinpath(datadir, "hab_qual_1000.asc"); missingval=NaN)
```

```julia
non_matches = xor.(isnan.(permeabilities), isnan.(qualities))
permeabilities[non_matches] .= 1e-20
qualities[non_matches] .= 1e-20;
rast = RasterStack((; qualities, permeabilities))
```

```julia
theta = 0.001
rsp = RSP(; theta)
```

```julia
tmp = zeros(length(qualities))
tmp[4300] = 1
plot(Raster(tmp, dims(rast)); title="Target (or is it Source?) Pixel")
```

# Step 2: Euclidean distances

```julia
euclid = solve(EuclidianDistance(), rsp, rast) 
plot(Raster(euclid[:, 4300], dims(qualities)); title="Euclidean Distance")
```

# Step 3: RSP Expected Cost Distances

```julia
dists = solve(ExpectedCost(), rsp, rast) 
```

```julia
plot(Raster(dists[:, 4300], dims(rast)); title="RSP Expected Cost Distance")
```

```julia
plot(euclid[:, 4300], dists[:, 4300]; 
    alpha=0.2,
    bordercolor=nothing,
    seriestype=:scatter, 
    legend=false, 
    xlabel="Euclidean Distance", 
    ylabel="Expected Cost"
)
```

```julia
target_dists = map(x -> exp(-x / 1000), dists[:, 4300])
plot(Raster(target_dists, dims(rast)); 
    title="RSP Expected Cost Proximity (log)"
)
```

```julia
target_dists = map(x -> x < 350 ? 1 : 350 / x, dists[:, 4300])
plot(Raster(target_dists, dims(rast)); 
    title="RSP Expected Cost Proximity (inv)"
)
```

```julia
plot(
    map(x -> exp(-x / 1000), dists[:, 4300]), 
    map(x -> x < 350 ? 1 : 350 / x, dists[:, 4300]); 
    alpha=0.2,
    seriestype=:scatter, 
    legend=false, 
    xlabel="log", 
    ylabel="inv"
)
```

# Step 4: Survival Proximity

```julia
surv_prob = solve(SurvivalProbability(), rsp, rast)
plot(Raster(surv_prob[:, 4300], dims(rast)); title="Survival proximity")
```

```julia
plot(euclid[:, 4300], surv_prob[:, 4300]; 
    seriestype=:scatter, 
    alpha=0.2,
    legend=false, 
    xlabel="Euclidean Distance", 
    ylabel="Surival Probability"
)
```

```julia
plot(dists[:, 4300], surv_prob[:, 4300]; 
    seriestype=:scatter, 
    alpha=0.2,
    legend=false, 
    xlabel="Expected Cost", 
    ylabel="Surival Probability"
)
```

```julia
plot(map(x -> exp(-x / 1000), dists[:, 4300]), surv_prob[:, 4300]; 
    seriestype=:scatter, 
    alpha=0.2,
    legend=false, 
    xlabel="Expected Cost Proximity", 
    ylabel="Surival Probability"
)
```
