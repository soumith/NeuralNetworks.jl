using NeuralNetworks

net = Serial(Float64, 1)
add_layer!(net, ReLU(Float64, 1))
add_layer!(net, ReLU(Float64, 1))

println(forward!(net, [1.0, 2.0, -3.0]))
println(backward!(net, [1.0, 2.0, -3.0], [10.0, 20.0, 30.0]))

