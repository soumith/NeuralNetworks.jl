type Module
    gradInput::Array{Float32}
    output::Array{Float32}
    
    function Module()
        println("Hello World") 
        new()
    end

    function updateOutput(input::Array{Float32})
    end

    function forward(input::Array{Float32})
        self
    end

    function updateGradInput(input::Array{Float32}, gradOutput::Array{Float32})
    end

    function accGradParameters(input::Array{Float32}, gradOutput::Array{Float32})
    end


end
