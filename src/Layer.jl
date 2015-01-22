abstract Layer{T<:Real}

abstract Container{T<:Real} <: Layer{T}

type Threshold{T<:Real} <: Layer{T}
    threshold :: T
    output :: Array{T}
end

Threshold{T<:Real}(t :: T) = Threshold{T}(t, Array(T))

