type Sigmoid{A<:AbstractArray} <: Pointwise{A}
    output :: A
    gradInput :: A
end

Sigmoid(T::Type, s...) = Sigmoid(Array(T, s...), Array(T, s...))

sigmoid(x) = 1 / (1 + exp(-x))
sigmoidPrime(x) = sigmoid(x) * (1 - sigmoid(x))

function updateOutput!{A}(self :: Sigmoid{A}, input :: A)
    map!(sigmoid, self.output, input)
    return self.output
end

function updateGradInput!{A}(self :: Sigmoid{A}, input :: A, gradOutput :: A)
    map!(sigmoidPrime, self.gradInput, gradOutput)
    return self.gradInput
end

export Sigmoid
    
