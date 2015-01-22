using NeuralNetworks

net = Serial(Float64, (10, 1), (3, 1))
add_layer!(net, Linear(Float64, 1, 10, 3))
add_layer!(net, ReLU(Float64, 3, 1))
validate(net)

initWeights!(net)

input = linspace(1.0, 10.0, 10)[:,:]
output = forward!(net, input)

gradOutput = linspace(1.0, 10.0, 3)[:,:]
gradInput = backward!(net, input, gradOutput)

println(output)

