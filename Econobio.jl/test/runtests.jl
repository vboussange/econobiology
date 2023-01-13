using Econobio, Test

@testset "Econobio" begin
    include("models.jl")
    include("composable_model.jl")
    include("result_econobio.jl")
    include("lsq_fit.jl")
    include("inference_model.jl")
end