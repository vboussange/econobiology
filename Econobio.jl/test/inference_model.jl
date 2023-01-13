#=
Using Turing on synthetic data
=#
cd(@__DIR__)
using MiniBatchInference, OrdinaryDiffEq, Econobio
using DiffEqSensitivity
using Optimization, OptimizationFlux, OptimizationOptimJL # for optimizers
using LinearAlgebra
using SimpleWeightedGraphs, LightGraphs
using Random; Random.seed!(1)
using Statistics

#=
Meta parameters
=#
@testset "Inference with MiniBatchInference" begin
    plotting = false
    alg = BS3()
    nruns = 5 # number of runs
    sensealg = DiffEqSensitivity.ForwardDiffSensitivity()
    optimizers = [ ADAM(1e-3), BFGS(initial_stepnorm=0.001)]
    epochs = [1000, 300]#could be set to 10000 to make sure that things converge
    group_size_init = 29
    continuity_term = 0.
    ic_term = 1/group_size_init
    verbose = true

    tsteps = 0:0.5:10
    tspan = (tsteps[1], tsteps[end])

    N = 10
    g = complete_graph(N)
    lap_mf = Matrix{Float32}(laplacian_matrix(g)) ./ Float32(N)

    u0_true = rand(N)
    p_true = [rand(N); rand(N); 0.1; 5e-3]
    mymodel = Modelαμ(ModelParams(N=N,lap=lap_mf, p = p_true), (Abs{0}(),Abs{0}(),Identity{0}(),Abs{0}()))

    prob = ODEProblem(mymodel, u0_true, tspan)
    sol = solve(prob, alg, saveat = tsteps, p = mymodel.mp.p)

    # using Plots
    # scatter(sol)
    odedata = Array(sol) .* exp.(0.2*rand(size(sol)...)) 
    # scatter(odedata')
    padding_data = odedata .> 0.

    callback = plotting ? (θs, p_trained, losses, pred, ranges) -> cb(θs, 
                                                                        p_trained, 
                                                                        losses, 
                                                                        pred, 
                                                                        ranges, 
                                                                        odedata, 
                                                                        tsteps,) : nothing    
    @time res = minibatch_MLE(group_size = group_size_init,
                                    optimizers = optimizers,
                                    p_init = p_true .* exp.(0.2 * rand(length(p_true))),
                                    data_set = odedata, 
                                    prob = prob, 
                                    tsteps = tsteps, 
                                    alg = alg, 
                                    sensealg = sensealg, 
                                    epochs = epochs,
                                    continuity_term = continuity_term,
                                    ic_term = ic_term,
                                    verbose = verbose,
                                    loss_fn = (data, pred, rg, ic_term) -> loss_log(data, pred, rg, ic_term, padding_data),
                                    cb = callback
                                    )

    reseco = ResultEconobio(mymodel,res)

    println("Inference <--> true params")
    [println(reseco.m.mp.p[i], " <--> ", p_true[i]) for i in 1:length(p_true)]
    @test mean(abs.(reseco.m.mp.p .- p_true) ./ p_true) < 5e-1
end