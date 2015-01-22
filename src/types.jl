abstract Layer

abstract Pointwise <: Layer

type Threshold
    # TODO: Define my fields here...
end

abstract Container <: Layer

type Serial <: Container
    # TODO: Define my fields here...
end

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
