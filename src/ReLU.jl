type ReLU{A<:AbstractArray} <: Pointwise{A}
    output :: A
    gradInput :: A
end

ReLU(T::Type, s...) = ReLU(Array(T, s...), Array(T, s...))

function updateOutput!{A}(self :: ReLU{A}, input :: A)
    self.output = input .* (input .>= zero(eltype(input)))
    return self.output
end

function updateGradInput!{A}(self :: ReLU{A}, input :: A, gradOutput :: A)
    self.gradInput = gradOutput .* (input .>= zero(eltype(input)))
    return self.gradInput
end

export ReLU
