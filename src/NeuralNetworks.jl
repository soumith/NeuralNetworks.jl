module NeuralNetworks
include("types.jl")
include("Shape.jl")
#include("SpatialConv.jl")

# weighted layers
include("Linear.jl")

# nonlinearities
include("ReLU.jl")
include("Sigmoid.jl")

# containers
include("Serial.jl")

# miscellaneous
include("View.jl")

# criterions
include("MSECriterion.jl")

end
