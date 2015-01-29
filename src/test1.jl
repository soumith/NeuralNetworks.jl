using NeuralNetworks

net = Serial(Float64, (10, 2, 1), (6, 1))
add_layer!(net, View(Float64, (10, 2, 1), (20, 1)))
add_layer!(net, Linear(Float64, 1, 20, 6))
add_layer!(net, Sigmoid(Float64, 6, 1))
validate(net)

initWeights!(net)

input = reshape(linspace(1.0, 10.0, 20), (10, 2, 1))
output = forward!(net, input)

println(output)

gradOutput = reshape(linspace(1.0, 10.0, 6), (6, 1))
gradInput = backward!(net, input, gradOutput)

println(gradInput)

