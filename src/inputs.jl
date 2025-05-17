
# Input types

abstract type InputType end
abstract type Permeability <: InputType end

abstract type Quality <: InputType end
struct SourceQuality <: Quality end
struct TargetQuality <: Quality end

struct Likelihood <: Permeability end
struct Cost <: Permeability end


# For sensitivity analysis
abstract type CostAndLikelihood <: Permeability end
struct CostToLikelihood <: CostAndLikelihood end
struct LikelihoodToCost <: CostAndLikelihood end
