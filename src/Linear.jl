type Linear{A<:AbstractArray} <: Pointwise{A}
    output :: A
    gradInput :: A
    weight :: A
end

Linear(T::Type, input_shape, output_shape) =
    Linear(Array(T, output_shape), Array(T, s), Array(T, s))

function updateOutput!{A}(self :: Linear{A}, input :: A)
    
