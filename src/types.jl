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

type SpatialConv <: Layer
    # TODO: Define my fields here...
end
