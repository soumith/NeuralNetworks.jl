function updateOutput!{T <: FloatingPoint}(
    layer::Layer,
    input::Array{T},
)
    error("updateOutput!(layer, input) not implemented abstractly")
end

function forward!{T <: FloatingPoint}(
    layer::Layer,
    input::Array{T},
)
    error("forward!(layer, input) not implemented abstractly")
end

function updateGradInput!{T <: FloatingPoint}(
    layer::Layer,
    input::Array{T},
    gradOutput::Array{T}
)
    error("updateGradInput!(layer, input, gradOutput) not implemented abstractly")
end

function accGradParameters!{T <: FloatingPoint}(
    layer::Layer,
    input::Array{T},
    gradOutput::Array{T},
)
    error("accGradParameters!(layer, input, gradOutput) not implemented abstractly")
end
