
```julia
using ConScape, Rasters, ArchGDAL, Plots
```

# Import data

```julia
datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
```

```julia
rast = RasterStack((; permeabilities=joinpath(datadir, "mov_prob_1000.asc"),
                      qualities= joinpath(datadir, "hab_qual_1000.asc")); missingval=NaN)
```

```julia
matches = boolmask(rast)
rast.permeabilities[Not(matches)] .= 1e-20
rast.qualities[Not(matches)] .= 1e-20;
```

# Amount of Connected Habitat

## Summed Expected Cost

Note: should be sum over both sources and targets

```julia
rsp = RSP(; theta=1.0, distance_transformation=ExpMinusAlpha(2000))
ch = solve(ConnectedHabitat(), rsp, rast)
```

```julia
plot(ch)
```

```julia
sum(t -> isnan(t) ? 0.0 : t, ch)
```

```julia
qˢ = [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list]
qᵗ = [h.g.target_qualities[i] for i in targetidx]
```

```julia
ec = solve(ExpectedCost(), rsp, rast)
similarities = map(t -> iszero(t) ? t : exp(-t / 2000), ec)
```

```julia
Raster(similarities[4300, :], dims(rast)) |> plot
```

```julia
ch1 = qˢ .* similarities * qᵗ

@time sum(ch1)
```

```julia
sum(t -> isnan(t) ? 0.0 : t, ch1)
```

```julia
Raster(ch1, dims(rast)) |> plot
```

```julia
landscape = qˢ .* similarities .* qᵗ'
```

```julia
@time sum(ch1)
```

## Eigenvalue Expected Cost

```julia
rsp = RandomisedShortestPath(; 
    proximity_measure=ExpectedCost(), 
    distance_transformation=t -> iszero(t) ? t : exp(-t / 2000),
    theta=1.0,
)
@time vˡ, λ, vʳ = solve(EigMax(), rsp, rast)
λ
```

```julia
Raster(real.(vʳ), dims(rast)) |> plot
```

```julia
Raster(abs.(real.(vˡ)), dims(rast)) |> plot
```

```julia
plot(ch1, real.(vʳ); seriestype=:scatter, legend=false, xlabel="sum", ylabel="eigenvector")
```

## Survival

```julia
similarities = h.Z;
```

```julia
Raster(similarities[4300, :], dims(rast)) |> plot
```

```julia
ch3 = qˢ .* similarities * qᵗ

@time sum(ch3)
```

```julia
Raster(ch3, dims(rast)) |> plot
```

```julia
plot(ch1, ch3; seriestype=:scatter, legend=false, xlabel="sum", ylabel="eigenvector")
```

## Probability of Connectivity

```julia
lcps = solve(ExpectedCost(), LeastCost(), rast)
Raster(lcps[4300, :], dims(rast)) |> plot
```

```julia
similarities = map(t -> iszero(t) ? t : exp(-t/2.5), lcps);
```

```julia
Raster(similarities[4300, :], dims(rast)) |> plot
```

```julia
pc = qˢ .* similarities * qᵗ

@time sum(pc)
```

```julia
Raster(pc, dims(rast)) |> plot
```
