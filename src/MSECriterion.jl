type MSECriterion{A <: AbstractArray, B <: AbstractArray} <: Criterion{A, B}
    output :: B
    gradInput :: A
end

MSECriterion(T :: Type, batch_size, input_size) =
    MSECriterion(Array(T, (batch_size,)), Array(T, (input_size, batch_size)))

function updateOutput!{A,B}(self :: MSECriterion{A,B}, input :: A, target :: A)
    local input_size = size(self.gradInput)[1]
    local batch_size = size(self.gradInput)[2]
    local sum = zero(eltype(A))
    for b = 1:batch_size
        sum = 0
        for i = 1:input_size
            local r = input[i,b] - target[i,b]
            sum += r * r
        end
        sum /= input_size
        self.output[b] = sum
    end
    return self.output
end

function updateGradInput!{A,B}(self :: MSECriterion{A,B},
                               input :: A,
                               target :: A)
    local input_size = size(self.gradInput)[1]
    local batch_size = size(self.gradInput)[2]
    local sum = zero(eltype(A))
    local d = eltype(A)(2) / input_size
    for b = 1:batch_size
        sum = 0
        for i = 1:input_size
            self.gradInput[i,b] = (input[i,b] - target[i,b]) * d
        end
    end
    return self.gradInput
end

export MSECriterion
