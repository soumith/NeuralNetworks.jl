type Linear{A<:AbstractArray} <: Layer{A, A}
    output :: A
    gradInput :: A
    weight :: A
    bias :: A
    gradWeight :: A
    gradBias :: A
end

# input and output are 1d
Linear(T::Type, batch_size, input_size, output_size) =
    Linear(Array(T, (output_size, batch_size)),
           Array(T, (input_size, batch_size)),
           Array(T, (output_size, input_size)),
           Array(T, (output_size, 1)),
           Array(T, (output_size, input_size)),
           Array(T, (output_size, 1)))

function updateOutput!{A}(self :: Linear{A}, input :: A)
    broadcast!(+, self.output, self.weight * input, self.bias)
    return self.output
end

function updateGradInput!{A}(self :: Linear{A}, input :: A,
                                gradOutput :: A)
    self.gradInput = transpose(self.weight) * gradOutput
    return self.gradInput
end

function accGradParameters!{A}(self :: Linear{A}, input :: A,
                                  gradOutput :: A, scale :: eltype(A))
    self.gradWeight = self.gradWeight + scale * gradOutput * transpose(input)
    self.gradBias = self.gradBias + scale * sum(gradOutput, 2)
end

export Linear
