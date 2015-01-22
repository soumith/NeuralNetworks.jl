using Base.Test
import NeuralNetworks
nn = NeuralNetworks

m = nn.SpatialConv(Float32,3,16,5,5,1,1)
input = rand(Float32, 3, 32, 32)
output = nn.forward!(m, input)

@test 1 == 1
