import Distributions

function updateOutput!{T <: FloatingPoint}(
    layer::SpatialConv,
    input::Array{T},
)
    layer.output = resize!(vec(layer.output), layer.nOutputPlane*(size(input,2)-layer.kH+1)*(size(input,3)-layer.kW+1))
    layer.output = reshape(layer.output,layer.nOutputPlane, (size(input,2)-layer.kH+1), (size(input,3)-layer.kW+1))
    print(size(layer.output))
    for i = 1:layer.nOutputPlane
        output_i = sub(layer.output, :, :, i)
        for j = 1:layer.nInputPlane
            input_i = sub(input, :, :, i)
            kernel_i = sub(layer.weight, i, j, :, :)
            output_i += conv2(input_i, kernel_i)
        end
    end
    return output
end

function forward!{T <: FloatingPoint}(
    layer::Layer,
    input::Array{T},
)
    return updateOutput!(layer, input)
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

function SpatialConv{T <: FloatingPoint}(
                                         ::Type{T},
                                         nInputPlane::Integer,
                                         nOutputPlane::Integer,
                                         kernelHeight::Integer,
                                         kernelWidth::Integer,
                                         strideHeight::Integer,
                                         strideWidth::Integer
                                         )
    obj = SpatialConv(
                      nInputPlane,
                      nOutputPlane,
                      kernelHeight,
                      kernelWidth,
                      strideHeight,
                      strideWidth,
                      Array(T, nOutputPlane, nInputPlane, kernelHeight, kernelWidth),
                      Array(T, nOutputPlane),
                      Array(T, nOutputPlane, nInputPlane, kernelHeight, kernelWidth),
                      Array(T, nOutputPlane),
    Array(T,1,1,1),
    Array(T,1,1,1)
                      )
    # fill weight and bias with a Uniform distribution normalized by fanin
    std = 1/sqrt(kernelWidth*kernelHeight*nInputPlane)
    rand!(Distributions.Uniform(-std, std), obj.weight)
    rand!(Distributions.Uniform(-std, std), obj.bias)
    return obj
end
