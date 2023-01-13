using OrdinaryDiffEq, Test, LightGraphs, SimpleWeightedGraphs
using Bijectors: Exp, inverse, Identity
using Random; Random.seed!(2)
N = 10
tspan = (0., 1.)
tsteps = range(tspan[1], tspan[end], length=10)
g = complete_graph(N)
lap_mf = Matrix{Float32}(laplacian_matrix(g)) ./ Float32(N)

dists_arr = [(Exp{0}(), Identity{0}()), 
                (Exp{0}(), Identity{0}(),Identity{0}()),
                (Exp{0}(), Identity{0}(),Identity{0}()),
                (Exp{0}(), Identity{0}(),Identity{0}()),
                (Exp{0}(), Identity{0}(),Identity{0}(),Identity{0}()),
                (Exp{0}(), Identity{0}(),Identity{0}(),Identity{0}()),
                (Exp{0}(), Identity{0}(),Identity{0}(),Identity{0}()),
                (Exp{0}(), Identity{0}(),Identity{0}(),Identity{0}(),Identity{0}())]
models = [ModelLog,Modelα,Modelμ,Modelδ,Modelαμ,Modelαδ,Modelμδ,Modelαμδ]
@testset "testing models $(models[i])" for i in 1:length(models)
    dudt_log = models[i](ModelParams(N=N,
                                    lap=lap_mf, 
                                    uworld = t -> zeros(N), 
                                    u0 = rand(N),
                                    tspan = tspan,
                                    alg = BS3()),
                                    dists_arr[i])
    sol = simulate(dudt_log, p = [rand(2*N); fill(0.1, length(dudt_log) - 2*N)])
    @test sol.retcode == :Success
end


@testset "testing bijections for $(models[i])" for i in 1:length(models)
    model = models[i](ModelParams(N=N,lap=lap_mf), dists_arr[i])
    p_true = rand(length(model))
    paraminv = inverse(model.mp.st)(p_true)
    @test all(paraminv |> model.mp.st .≈ p_true)
end

using MiniBatchInference, LinearAlgebra
# TODO: make sure that the results are coherent
@testset "loglikelihood for $(models[i])" for i in 1:length(models)
    p_true = [rand(2*N); 0.1; 0.1; 0.1] # all coefficients not used by all model
    dudt_log = models[i](ModelParams(N=N,
                    lap=lap_mf, 
                    uworld = t -> zeros(N), 
                    u0 = rand(N),
                    tspan = tspan,
                    alg = BS3(),
                    kwargs_sol = Dict(:saveat => tsteps,) ),
                    dists_arr[i])
    ode_data = simulate(dudt_log, p = [rand(2*N); fill(0.1, length(dudt_log) - 2*N)]) |> Array

    group_size = 6
    ranges = get_ranges(group_size, length(tsteps))

    p_estimated = p_true .+ 0.02
    u0s = [ode_data[:,rg[1]] for rg in ranges]
    res = ResultEconobio(dudt_log, ResultMLE(p_trained = p_estimated[1:length(dudt_log)],
                                            ranges=ranges))
    @test Econobio.loglikelihood(res, ode_data, 0.1; u0s = u0s) < 1.0
end