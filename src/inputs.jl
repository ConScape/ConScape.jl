
# Input types

abstract type InputType end
abstract type Permeability <: InputType end

abstract type Quality <: InputType end
struct SourceQuality <: Quality end
struct TargetQuality <: Quality end

struct StepLikelihood <: Permeability end
struct StepCost <: Permeability end


# For sensitivity analysis
abstract type StepCostAndLikelihood <: Permeability end
struct StepCostToLikelihood <: StepCostAndLikelihood end
struct StepLikelihoodToCost <: StepCostAndLikelihood end
