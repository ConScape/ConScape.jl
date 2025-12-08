# TODO: docs
struct LandscapeMatrix <: GraphMeasure end

# TODO its not sparse
computelevel(::LandscapeMatrix) = TargetLevel()
returntrait(::LandscapeMatrix) = ReturnAssignedDense()

# This differs form FunctionalHabitat in that it returns the full size matrix
compute_target(::LandscapeMatrix, ti::TargetInit{<:Union{RSP,RandomWalk,LCP}}) = ti.M
