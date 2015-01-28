type Serial{A<:AbstractArray, B<:AbstractArray} <: Container{A,B}
    output :: B
    gradInput :: A
    modules :: Array{Layer, 1}
end

Serial(T::Type, input_size, output_size) =
    Serial(Array(T, output_size),
           Array(T, input_size),
           Array(Layer, 0))

function add_layer!{A, B}(self :: Serial{A, B}, layer :: Layer)
    if length(self.modules) == 0
        @assert(shape(self.gradInput) == shape(layer.gradInput),
                "Layer shape mismatch")
    else
        @assert(shape(last(self.modules).output) == shape(layer.gradInput),
                "Layer shape mismatch")
    end
    push!(self.modules, layer)
end

function validate{A, B}(self :: Serial{A, B})
    local input_shape = shape(self.gradInput)

    for layer in self.modules
        @assert(shape(layer.gradInput) == input_shape,
                "Layer shape mismatch")
        input_shape = shape(layer.output)
    end

    @assert(shape(self.output) == input_shape, "Layer shape mismatch")
end

function updateOutput!{A, B}(self :: Serial{A, B}, input :: A)
    for i = 1:length(self.modules)
        input = updateOutput!(self.modules[i], input)
    end
    self.output = input
    return self.output
end

function updateGradInput!{A, B}(self :: Serial{A, B},
                                input :: A,
                                gradOutput :: B)
    if !isempty(self.modules)
        for i = length(self.modules):-1:2
            gradOutput = updateGradInput!(self.modules[i],
                                          self.modules[i-1].output,
                                          gradOutput)
        end

        gradOutput = updateGradInput!(self.modules[1], input, gradOutput)
    end

    self.gradInput = gradOutput
    return self.gradInput
end

function accGradParameters!{A, B}(self :: Serial{A, B},
                                  input :: A,
                                  gradOutput :: B,
                                  scale :: eltype(A))
    if isempty(self.modules)
        return
    end

    for i = length(self.modules):-1:2
        layer = self.modules[i]
        accGradParameters!(layer,
                           self.modules[i-1].output,
                           gradOutput,
                           scale)
        gradOutput = layer.gradInput
    end

    accGradParameters!(self.modules[1], input, gradOutput, scale)
end

export Serial
