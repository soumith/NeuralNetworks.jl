include("Shape.jl")

a = Array(Float32, 1, 2)

function matches(s :: Shape, x)
    return s == shape(x)
end

@assert(matches(shape(a), Array(Float32, 1, 2)))
@assert(!matches(shape(a), Array(Int32, 1, 2)))
@assert(!matches(shape(a), Array(Float32, 1, 3)))
@assert(!matches(shape(a), Array(Float32, 1)))
@assert(!matches(shape(a), Array(Float32, 1, 2, 3)))


s = (Array(Float32, 1, 2), Array(Int32, 1, 3))

@assert(!matches(shape(a), s))
@assert(!matches(shape(s), a))
@assert(matches(shape(s), (Array(Float32, 1, 2), Array(Int32, 1, 3))))
@assert(!matches(shape(s), (Array(Float32, 1, 2),)))
@assert(!matches(shape(s), (Array(Float32, 1, 2), Array(Int32, 1, 3),
                            Array(Int32, 1, 4))))
@assert(!matches(shape(s), (Array(Float32, 1, 3), Array(Int32, 1, 3))))
@assert(!matches(shape(s), (Array(Float32, 1, 2), Array(Int32, 1, 2))))

x = 1.0

@assert(matches(shape(x), 2.0))
@assert(!matches(shape(x), a))
@assert(!matches(shape(x), s))
@assert(!matches(shape(x), 7))
