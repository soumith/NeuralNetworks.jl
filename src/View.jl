type View{A<:AbstractArray, B<:AbstractArray} <: Layer{A,B}
    output :: B
    gradInput :: A
end

function View(T::Type, input_size, output_size)
    @assert(prod(input_size) == prod(output_size),
            "View input / output size mismatch")
    return View(Array(T, output_size), Array(T, input_size))
end

function updateOutput!{A, B}(self :: View{A, B}, input :: A)
    self.output[:] = reshape(input, size(self.output))
    return self.output
end

function updateGradInput!{A, B}(self :: View{A, B},
                                input :: A,
                                gradOutput :: B)
    self.gradInput[:] = reshape(gradOutput, size(self.gradInput))
    return self.gradInput
end

export View
