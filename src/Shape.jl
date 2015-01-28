abstract Shape

immutable SequenceShape <: Shape
    children :: Array{Shape, 1}
end

function ==(a :: SequenceShape, b :: SequenceShape)
    return a.children == b.children
end

import Base.hash

const SequenceShape_seed = hash("SequenceShape")
function hash(a :: SequenceShape, h :: UInt)
    return hash(a.children, h + SequenceShape_seed)
end

function is_sequence(obj)
    if isa(obj, Tuple)
        return true
    end

    if !isa(obj, AbstractArray) || ndims(obj) != 1
        return false
    end

    if eltype(obj) <: Number
        return false
    end

    return true
end

immutable TensorShape <: Shape
    eltype :: Type
    size :: Array{Int64, 1}
end

function ==(a :: TensorShape, b :: TensorShape)
    return a.eltype == b.eltype && a.size == b.size
end

const TensorShape_seed = hash("TensorShape")
function hash(a :: TensorShape, h :: UInt)
    return hash(a.eltype, hash(a.size, h + TensorShape_seed))
end

function TensorShape(t :: Type, sizes :: Tuple)
    return TensorShape(t, collect(sizes))
end

function is_tensor(obj)
    return isa(obj, AbstractArray) && eltype(obj) <: Number
end

immutable TypeShape{T} <: Shape
end

function shape(obj)
    if is_sequence(obj)
        return SequenceShape(collect(map(shape, obj)))
    end
    if is_tensor(obj)
        return TensorShape(eltype(obj), collect(size(obj)))
    end
    return TypeShape{typeof(obj)}()
end

export shape
