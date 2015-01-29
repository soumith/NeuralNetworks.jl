using NeuralNetworks

net = Serial(Float64, (28 * 28, 1), (10, 1))
add_layer!(net, Linear(Float64, 1, 28 * 28, 300))
add_layer!(net, Sigmoid(Float64, (300, 1)))
add_layer!(net, Linear(Float64, 1, 300, 10))
add_layer!(net, Sigmoid(Float64, (10, 1)))
validate(net)

initWeights!(net)

const num_iters = 1000
const num_images = 60000

using MNIST

target = zeros(Float64, 10, 1)
criterion = MSECriterion(Float64, 1, 10)

for iter = 1:num_iters
    local loss
    loss = 0.0
    for img = 1:num_images
        local image_data = reshape(MNIST.trainfeatures(img), 28 * 28, 1)
        local label = Int(MNIST.trainlabel(img)) + 1
        target[label,1] = 1.0

        local out = forward!(net, image_data)
        loss += forward!(criterion, out, target)[1]
        local gi = backward!(criterion, out, target)
        zeroGradParameters!(net)
        backward!(net, image_data, gi)

        target[label,1] = 0.0
    end
    loss /= num_images
    println(iter, " ", loss)
end
