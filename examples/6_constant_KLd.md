```julia
using ConScape
using Rasters
using Optim
using Plots
```

TODO: remove this file! or describe it better its cool

# Import data

```julia
datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
```

```julia
permeabilities = Raster(joinpath(datadir, "mov_prob_1000.asc"); missingval=NaN)
qualities = Raster(joinpath(datadir, "hab_qual_1000.asc"); missingval=NaN)
```

```julia
non_matches = xor.(isnan.(permeabilities), isnan.(qualities))
permeabilities[non_matches] .= 1e-20
qualities[non_matches] .= 1e-20;
```

```julia
rast = RasterStack((; permeabilities, qualities))
```

# Set fixed KL-divergence

```julia
max_kld = solve(KullbackLeiblerDivergence(), LeastCost(), rast)
```

```julia
target_mean_kld = 0.5 * max_kld
```

Define a function to optimize:

```julia
function mean_kld_diff(theta, rast, target_mean_kld)
    mean_kld = solve(KullbackLeiblerDivergence(), RSP(; theta), rast)
    return (mean_kld - target_mean_kld)^2
end
```

Find theta with the specified KL divergence.
(This may take a while to run!)

```julia
r = optimize(0.0, 3.0; iterations=20) do theta 
    println("running for theta: $theta")
    mean_kld_diff(theta, rast, target_mean_kld) 
end
```

```julia
sqrt(r.minimum) / target_mean_kld
```

```julia
optimized_theta = r.minimizer
```

```julia
rsp_optimal = RSP(; theta=optimized_theta);
solve(KullbackLeiblerDivergence(), rsp_optimal, rast)
```

```julia
target_mean_kld
```

# Corridor example with constant KLd

```julia
nrows, ncols = 21, 41
# Qualities decrease by row
qualities = collect(reshape(collect(nrows * ncols:-1:1), ncols, nrows)')
g = ConScape.perm_wall_sim(nrows, ncols;
    qualities,
    corridorwidths=(3,),
    corridorpositions=(0.5,),
    impossible_affinity=0.0,
)
rsp = RSP(; theta=10.0)
```

```julia
max_kld = solve(KullbackLeiblerDivergence(), LeastCost(), rast);
```

```julia
100 * solve(KullbackLeiblerDivergence(), rsp, rast) / max_kld
```

```julia
target_node = Int(ceil((ncols - 1)/2 - 3) * nrows + ceil((nrows + 1) / 2))
dist = solve(ExpectedCost(), rsp, g);
```

```julia
plot(Raster(dist[:, target_node], dims(rast)))
```

```julia
target_kld = ConScape.mean_kl_divergence(h)
100 * solve(KullbackLeiblerDivergence(), rsp, rast) / max_kld
```

```julia
# Qualities decrease by row
qualities = copy(reshape(collect(nrows * ncols:-1:1), ncols, nrows)')
g2 = ConScape.perm_wall_sim(nrows, ncols;
    corridorwidths=(7,),
    corridorpositions=(0.5,),
    impossible_affinity=0.,
    qualities,
)
```

```julia
dist2 = solve(ExpectedCost(), rsp, g2)
```

```julia
solve(KullbackLeiblerDivergence(), rsp, rast) / max_kld
```

```julia
target_kld
```

```julia
r2 = optimize(0.0, 3.0; iterations=20) do theta
    mean_kld_diff(theta, g2, target_kld) 
end
```

```julia
sqrt(r2.minimum) / target_kld
```

```julia
opimized_theta = r2.minimizer
```

```julia
rsp3 = RSP(; theta=optimized_theta)
```

```julia
dist3 = solve(ExpectedCost(), rsp3, g2)
```

```julia
values_orig = ConScape.plot_values(g, map(t -> exp(-t / 0.5), dist[:,target_node]))
Plots.contour(values_orig; fill=true, levels=10, c=:plasma)
```

```julia
values_wide = ConScape.plot_values(g2, map(t -> exp(-t / 0.5), dist2[:,target_node]))
Plots.contour(values_wide; fill=true, levels=10, c=:plasma)
```

```julia
values_consKL = ConScape.plot_values(g2, map(t -> exp(-t / 0.5), dist3[:,target_node]))
Plots.contour(values_consKL; fill=true, levels=10, c=:plasma)
```