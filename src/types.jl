abstract Layer{A<:AbstractArray}

abstract Pointwise{A<:AbstractArray} <: Layer{A}

abstract Container{A<:AbstractArray} <: Layer{A}

function accGradParameters!{A}(self :: Container{A}, input :: A,
                               gradOutput :: A, scale :: eltype(A))
end

function accGradParameters!{A}(self :: Container{A}, input :: A,
                               gradOutput :: A)
    return accGradParameters!(self, input, one(eltype(A)))
end

function forward!{A}(self :: Layer{A}, input :: A)
    return updateOutput!(self, input)
end
export forward!

function backward!{A}(self :: Layer{A}, input :: A, gradOutput :: A,
                      scale :: eltype(A))
    updateGradInput!(self, input, gradOutput)
    accGradParameters!(self, input, gradOutput, scale)
    return self.gradInput
end

function backward!{A}(self :: Layer{A}, input :: A, gradOutput :: A)
    return backward!(self, input, gradOutput, one(eltype(A)))
end
export backward!

function add_layer!{A}(self :: Container{A}, layer :: Layer{A})
    push!(self.modules, layer)
end
export add_layer!

#######

type Parallel <: Container
    # TODO: Define my fields here...
end

type SoftMax <: Layer
    # TODO: Define my fields here...
end

type LogSoftMax <: Layer
    # TODO: Define my fields here...
end

type Affine <: Layer
    # TODO: Define my fields here...
end

type SpatialMaxPool <: Layer
    # TODO: Define my fields here...
end

type SpatialConv{T, I <: Integer} <: Layer
    nInputPlane::I
    nOutputPlane::I
    kH::I
    kW::I
    dH::I
    dW::I
    weight::Array{T}
    bias::Array{T}
    grad_weight::Array{T}
    grad_bias::Array{T}
    output::Array{T}
    gradInput::Array{T}
end
