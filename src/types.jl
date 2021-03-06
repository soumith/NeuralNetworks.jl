abstract Layer{A<:AbstractArray, B<:AbstractArray}

# Pointwise layers always use the same dimensionality for input and output
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

# Add a layer to the container
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

# state is (0, false) for initial setup (must return self next)
#          (idx, state_for_modules[idx]) afterwards
function start(self :: Container)
    return 0, false
end

function done(self :: Container, state)
    local idx, o_state
    idx, o_state = state
    if idx == 0 then
        return false
    end
    if idx > length(self.modules)
        return true
    end

    # Note that this check is simpler than in the general case, as we know
    # each layer will return at least one element (self). So, as long
    # as idx < length(self.modules), we know we're not done; this would no
    # longer be true if we allowed empty layers.
    return done(self.modules[idx], o_state) && idx == length(self.modules)
end

function next(self :: Container, state)
    local idx, o_state
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
                local a = getfield(layer, field)
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

function getParameters(self :: Layer)
    local params = Array((AbstractArray, AbstractArray), 0)
    for layer in self
        if :weight in names(layer) && :gradWeight in names(layer)
            push!(params, (layer.weight, layer.gradWeight))
        end
        if :bias in names(layer) && :gradBias in names(layer)
            push!(params, (layer.bias, layer.gradBias))
        end
    end
    return params
end
export getParameters
            
abstract Criterion{A<:AbstractArray, B<:AbstractArray}

function forward!{A,B}(c :: Criterion{A,B}, x :: A, t :: A)
    return updateOutput!(c, x, t)
end

function backward!{A,B}(c :: Criterion{A,B}, x :: A, t :: A)
    return updateGradInput!(c, x, t)
end


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
