abstract Layer{A<:AbstractArray, B<:AbstractArray}

abstract Pointwise{A<:AbstractArray} <: Layer{A, A}

abstract Container{A<:AbstractArray, B<:AbstractArray} <: Layer{A, B}

# By default, accGradParameters! does nothing
function accGradParameters!{A, B}(self :: Layer{A, B}, input :: A,
                                  gradOutput :: B, scale :: eltype(A))
end

# Call the scale version if scale unspecified
function accGradParameters!{A, B}(self :: Layer{A, B}, input :: A,
                                  gradOutput :: B)
    return accGradParameters!(self, input, one(eltype(A)))
end

function forward!{A, B}(self :: Layer{A, B}, input :: A)
    return updateOutput!(self, input)
end
export forward!

function backward!{A, B}(self :: Layer{A, B}, input :: A, gradOutput :: B,
                         scale :: eltype(A))
    updateGradInput!(self, input, gradOutput)
    accGradParameters!(self, input, gradOutput, scale)
    return self.gradInput
end

# Call the scale version if scale unspecified
function backward!{A, B}(self :: Layer{A, B}, input :: A, gradOutput :: B)
    return backward!(self, input, gradOutput, one(eltype(A)))
end
export backward!

function add_layer!{A, B}(self :: Container{A, B}, layer :: Layer)
    push!(self.modules, layer)
end
export add_layer!

function validate{A, B}(self :: Layer{A, B})
end

function validate{A, B}(self :: Container{A, B})
    for layer in self.modules
        validate(layer)
    end
end
export validate

# Iteration; layers return themselves; containers return themselves and
# children

import Base.start
import Base.done
import Base.next

function start(self :: Layer)
    return false
end

function done(self :: Layer, state)
    return state
end

function next(self :: Layer, state)
    return self, true
end

function start(self :: Container)
    return 0, false
end

function done(self :: Container, state)
    idx, o_state = state
    if idx == 0 then
        return false
    end
    if idx > length(self.modules)
        return true
    end

    return done(self.modules[idx], o_state) && idx == length(self.modules)
end

function next(self :: Container, state)
    idx, o_state = state
    if idx == 0
        return self, (1, isempty(self.modules) ? true : start(self.modules[1]))
    end

    if done(self.modules[idx], o_state)
        idx += 1
        o_state = start(self.modules[idx])
    end

    item, o_state = next(self.modules[idx], o_state)
    return item, (idx, o_state)
end

function zeroGradParameters!(self :: Layer)
    for layer in self
        for field in (:gradWeight, :gradBias)
            if field in names(layer)
                a = getfield(layer, field)
                fill!(a, zero(eltype(a)))
            end
        end
    end
end
export zeroGradParameters!

using Distributions
function initWeights!(self :: Layer)
    for layer in self
        if :weight in names(layer)
            rand!(Distributions.Uniform(-0.1, 0.1), layer.weight)
        end
        if :bias in names(layer)
            rand!(Distributions.Uniform(0, 0.1), layer.bias)
        end
    end
end
export initWeights!
            

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
