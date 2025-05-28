# Landmarks: coarse graining

This example demonstrates thinning of target qualities to 
create "landmark" targets at reduced resolution. 

It will demonstrate correlation between using all targets
and coarse grained targets of varying density, for both
connectivity and betweeness measure.

```julia
using ConScape
using Rasters
using ArchGDAL
using LinearAlgebra
using Plots
using Statistics
```

# Data import

```julia
datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
permeabilities = Raster(joinpath(datadir, "mov_prob_1000.asc"); missingval=NaN)
qualities = Raster(joinpath(datadir, "hab_qual_1000.asc"); missingval=NaN)
```

Cleanup

```julia
non_matches = xor.(isnan.(permeabilities), isnan.(qualities))
permeabilities[non_matches] .= 1e-20
qualities[non_matches] .= 1e-20;
```

```julia
rast = RasterStack((; permeabilities, qualities))
```

# Coarse graining

`ConScapeProblem` has a `grain` keyword that triggers coarse graining
when the `target_qualities` data is loaded from a `RasterStack`.

Here we define a problem that uses it, and one that doesn't.

```julia
theta = 0.01
grain = 20
rsp = RandomisedShortestPath(; theta, distance_transformation=x -> exp(-x/75))
```

We can solve the connected habitat for both:

```julia
ch = @time solve(ConnectedHabitat(), rsp, rast)
ch_coarse = @time solve(ConnectedHabitat(), rsp, rast; grain)
```

And compare the sum:

```julia
ch_sum = sum(t -> isnan(t) ? 0.0 : t, ch)
ch_sum_coarse = sum(t -> isnan(t) ? 0.0 : t, ch_coarse)
ch_sum_coarse / ch_sum 
```

And corelation

```julia
cor(filter(!isnan, ch), filter(!isnan, ch_coarse))
```

Then do the same for betweenness:

```julia
betq = solve(Betweenness(QualityWeighted()), rsp, rast)
plot(betq)
```

```julia
betq_coarse = solve(Betweenness(QualityWeighted()), rsp, rast; grain)
plot(betq_coarse)
```

```julia
cor(filter(!isnan, betq), filter(!isnan, betq_coarse))
```

```julia
betm = @time solve(Betweenness(QualityAndProximityWeighted()), rsp, rast)
betm_coarse = @time solve(Betweenness(QualityAndProximityWeighted()), rsp, rast; grain)
```

```julia
cor(filter(!isnan, betm), filter(!isnan, betm_coarse))
```

We can write this whole comparison using `ConScapeProblem` definitions:

```julia
measures = (;
    ch=ConScape.ConnectedHabitat(),
    betq=ConScape.Betweenness(QualityWeighted()),
    betm=ConScape.Betweenness(QualityAndProximityWeighted()),
)
problem = ConScapeProblem(; measures, movement)
problem_coarse = ConScapeProblem(; measures, movement, grain)
```

Then run it for all operations on both normal and coarse grids

```julia
rast = RasterStack((; permeabilities, qualities))
result = ConScape.solve(problem, rast)
result_coarse = ConScape.solve(problem_coarse, rast)
```

And plot these outputs:

```julia
plot(result)
```

```julia
plot(result_coarse)
```

Coarse graining also works in `WindowedProblem`, and is
applied as each window is loaded from the full raster.

```julia
windowed_problem = WindowedProblem(problem_coarse; 
    centersize=10, buffer=20, threaded=false
)
@time result = solve(windowed_problem, rast)
plot(result)
```

And we can check the corelation similarly to above, by getting
layers from `result` and `result_coarse`

```julia
cor(collect(skipmissing(result.betq)), collect(skipmissing(result_coarse.betq)))
cor(collect(skipmissing(result.betm)), collect(skipmissing(result_coarse.betm)))
cor(collect(skipmissing(result.ch)),   collect(skipmissing(result_coarse.ch)))
```

# Test landmark performance for amount of connected habitat

Calculate connected habitat at varying coarseness:

```julia
coarseness = (1, 2, 3, 5, 7, 10, 15, 20)
ch_by_grain = map(1:length(coarseness)) do grain
    ConScape.solve(ConnectedHabitat(), rsp, rast; grain)
end
```

Then calculate the sums

```julia
est_ch = map(ch_c -> sum(t -> isnan(t) ? 0.0 : t, ch_c), ch_by_grain)
```

And correlations with the original connected habitat

```julia
cor_ch = map(ch_c -> cor(filter(!isnan, ch), filter(!isnan, ch_c)), ch_by_grain)
```

```julia
sum_ch = sum(t -> isnan(t) ? 0.0 : t, ch)
plot(est_ch / sum_ch)
```

```julia
est_ch / sum_ch
```

```julia
plot(cor_ch)
```

```julia
cor_ch
```

# Test landmark performance for betweenness

## Quality weighted

```julia
cor_betq = map(1:length(coarseness)) do grain
    betq_coarse = solve(Betweenness(QualityWeighted()), rsp, rast; grain)
    return cor(filter(!isnan, betq), filter(!isnan, betq_coarse))
end
```

```julia
plot(cor_betq)
```

## Proximity weighted

```julia
cor_betm = map(1:length(coarseness)) do grain
    betm_coarse = solve(Betweenness(QualityAndProximityWeighted()), rsp, rast; grain)
    return cor(filter(!isnan, betm), filter(!isnan, betm_coarse))
end
```

```julia
plot(cor_betm)
```

# Figures for the paper

```julia
plot([cor_ch, cor_betq, cor_betm];  
    title = "Correlation",
    xlabel = "Coarseness",
    xticks = (1:8, coarseness),
    ylims = (0.75, 1.0),
    label = ["Amount Connected Habitat" "Quality-weighted Betweenness" "Quality-and-proximity-weighted Betweenness"],
    legend=:bottomleft,
    lw = 3
)
#savefig("output_figures/figure_lm_cors.png")
```

```julia
plot(est_ch / sum_ch; 
    title = "Estimated Amount Connected Habitat",
    xlabel = "Coarseness",
    xticks = (1:8, coarseness),
    legend=false,
    lw = 3
)
#savefig("output_figures/figure_lm_conhab.png")
```

```julia
plot((est_ch .- sum_ch) / sum_ch * 100;  
    title = "% Underestimation of the landscape's Connected Habitat",
    xlabel = "Coarseness",
    xticks = (1:8, coarseness),
    legend = false,
    lw = 3
)
# savefig("output_figures/figure_lm_conhab.png")
```

```julia
(est_ch .- sum_ch) / sum_ch * 100
```
