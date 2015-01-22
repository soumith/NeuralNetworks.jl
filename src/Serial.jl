type Serial{A<:AbstractArray} <: Container{A}
    output :: A
    gradInput :: A
    modules :: Array{Layer{A}, 1}
end

Serial(T::Type, s...) = Serial(Array(T, s), Array(T, s),
                               Array(Layer{Array{T, length(s)}}, 0))

function updateOutput!{A}(self :: Serial{A},
                          input :: A)
    for i = 1:length(self.modules)
        input = updateOutput!(self.modules[i], input)
    end
    self.output = input
    return self.output
end

function updateGradInput!{A}(self :: Serial{A},
                             input :: A,
                             gradOutput :: A)
    self.gradInput = gradOutput
    
    for i = length(self.modules):-1:2
        gradOutput = updateGradInput!(self.modules[i],
                                      self.modules[i-1].output,
                                      gradOutput)
    end

    if length(self.modules) > 0
        gradOutput = updateGradInput!(self.modules[1], input, gradOutput)
    end

    self.gradInput = gradOutput
    return self.gradInput
end

function accGradParameters!{A}(self :: Serial{A},
                               input :: A,
                               gradOutput :: A,
                               scale :: eltype(A))
    for i = length(self.modules):-1:2
        layer = self.modules[i]
        accGradParameters!(layer,
                           self.modules[i-1].output,
                           gradOutput,
                           scale)
        gradOutput = layer.gradInput
    end

    if length(self.modules) > 0
        accGradParameters!(self.modules[1], input, gradOutput, scale)
    end
end

export Serial
