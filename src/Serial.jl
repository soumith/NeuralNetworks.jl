type Serial{A<:AbstractArray, B<:AbstractArray} <: Container{A,B}
    output :: B
    gradInput :: A
    modules :: Array{Layer, 1}
end

Serial(T::Type, input_size, output_size) =
  Serial(Array(T, output_size), Array(T, input_size),
  Array(Layer, 0))

function add_layer!{A, B}(self :: Serial{A, B}, layer :: Layer)
    if length(self.modules) == 0
        @assert(typeof(self.gradInput) == A, "Layer type mismatch")
        @assert(size(self.gradInput) == size(layer.gradInput),
                "Layer size mismatch")
    else
        @assert(typeof(last(self.modules).output) == A, "Layer type mismatch")
        @assert(size(last(self.modules).output) == size(layer.gradInput),
                "Layer size mismatch")
    end
    push!(self.modules, layer)
end

function validate{A, B}(self :: Serial{A, B})
    input_type = A
    input_size = size(self.gradInput)
    for layer in self.modules
        @assert(typeof(layer.gradInput) == input_type, "Layer type mismatch")
        @assert(size(layer.gradInput) == input_size, "Layer size mismatch")
        validate(layer)
        input_type = typeof(layer.output)
        input_size = size(layer.output)
    end
    @assert(typeof(self.output) == input_type, "Layer type mismatch")
    @assert(size(self.output) == input_size, "Layer size mismatch")
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

function accGradParameters!{A, B}(self :: Serial{A, B},
                                  input :: A,
                                  gradOutput :: B,
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
