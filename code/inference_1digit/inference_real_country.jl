#=
utilities functions used for single simulations and multi country simulations
* Version 
- v2:
    - NegAbs
    - initialisation with functions
    - multiple runs
=#
cd(@__DIR__)
using Interpolations
using MiniBatchInference, OrdinaryDiffEq, DiffEqSensitivity, Econobio
using LightGraphs, SimpleWeightedGraphs
using DataFrames, JLD2, UnPack, Dates
using LinearAlgebra
# using Random; Random.seed!(3);
using Bijectors: Identity, inverse
# using PyPlot
using Optimization, OptimizationOptimJL, OptimizationFlux, LineSearches
using ProgressMeter
using Distributions: LogNormal

function inference(;kwargs...) 
    @unpack ranges, model, odedata, padding_data, optims, ic_term, prior_scaling, continuity_term = kwargs
    N = size(odedata,1)
    prob = get_prob(model, u0 = model.mp.u0[1:N])
    p_init = prob.p # taking params from prob allow to get the inversed params
    tsteps = model.mp.kwargs_sol[:saveat]
    verbose =  model.mp.kwargs_sol[:verbose]
    callback = plotting ? (θs, p_trained, losses, pred, ranges) -> cb(θs, 
                                                                        p_trained, 
                                                                        losses, 
                                                                        pred, 
                                                                        ranges, 
                                                                        odedata, 
                                                                        tsteps,) : nothing

    # making sure that the ODE solver works with the initial parameters
    sol = solve(prob, alg, saveat = tsteps, p =  inverse(model.mp.st)(p_init))
    @assert (sol.retcode == :Success && sol.retcode !== :Terminated) "Initial parameters are not realistic"

    @time res = MiniBatchInference._minibatch_MLE(ranges = ranges,
                                                optimizers = optims,
                                                p_init = inverse(model.mp.st)(p_init), # transforming parameters 
                                                data_set = odedata, 
                                                prob = prob, 
                                                tsteps = tsteps, 
                                                alg = alg, 
                                                sensealg = sensealg, 
                                                epochs = epochs,
                                                continuity_term = continuity_term,
                                                ic_term = ic_term,
                                                verbose_loss = verbose_loss,
                                                loss_fn = (data, p, pred, rg, ic_term) -> loss_log(data, 
                                                                                                p,
                                                                                                pred, 
                                                                                                rg, 
                                                                                                ic_term, 
                                                                                                padding_data,
                                                                                                prior_scaling),
                                                cb = callback,
                                                save_pred = true, # required for AIC test
                                                info_per_its = info_per_its,
                                                maxiters=100, 
                                                u0s_init = model.mp.u0,
                                                verbose = verbose
                                                # dtmin = 1e-5# number of iterations of the ODE solver
                                                )

    reseco = construct_result(model,res)

    return reseco
end

function simu(pars)
    @unpack model, country, odedata = pars
    println("Optimizing for ", model, ", $country")

    if plotting
        fig, ax = subplots()
        ax.plot(odedata', linestyle = ":")
        ax.set_yscale("log")
        display(fig)
    end

    res = inference(;pars...)
    # calculating Statistics
    # NOTE: to calculate likelihood ratio, we do not need to have σ estimated (difference of likelihood), so we set it to 0
    σ = estimate_σ(res, odedata, noisedistrib = LogNormal())
    l = Econobio.loglikelihood(res, odedata, σ; loglike_fn = MiniBatchInference.loglikelihood_lognormal)
    aic = AIC(l, length(model))
    return (name(model), res, l, aic, Inf, country)
end
                    
#=
Initialising logistic model parameters, used for all the models
=#
function initialize_ModelLog(odedata, ranges, tspan, tsteps, uworld)
    N = size(odedata,1)
    u0s_init = odedata[:,first.(ranges)]
    minu0s = minimum(x -> iszero(x) ? Inf : x, odedata, dims=2)
    for j in 1:length(ranges)
        for i in 1:N
            if iszero(u0s_init[i,j])
                u0s_init[i,j] = minu0s[i] # replacing value with minimum known value to have plausible results
            end
        end
    end
    # _, rinit, binit, RSS = init_r_b_u0(odedata,tsteps)
    u0s_init = reshape(u0s_init,:)
    rinit = 1e-1 .* (rand(N) .+ 0.5)
    binit = 1e0 .* (rand(N) .+ 0.5)
    Dists = (Abs{0}(), Abs{0}())
    g = complete_graph(N)
    lap_mf = Matrix{Float64}(laplacian_matrix(g)) ./ Float64(N)

    Model = ModelLog
    p_init = [rinit; binit;]
    model = Model(ModelParams(N=N,
                        lap = lap_mf,
                        uworld = uworld,
                        p = p_init,
                        u0 = u0s_init,
                        alg=alg,
                        tspan=tspan,
                        kwargs_sol = Dict(:saveat => tsteps, :verbose => false)),
                        Dists)
    return model
end
# helper function to run a loop with initialize methods
initialize_ModelLog(model) = model

function initialize_Modelαp(m::ModelLog, init_p = 1.)
    p_init = [m.mp.p; init_p * (rand() + 0.5)]
    mp = remake(m.mp, p = p_init)
    Dists = (Abs{0}(), Abs{0}(), Abs{0}())
    model = Modelα(mp,Dists)
    return model
end

function initialize_Modelαn(m::ModelLog, init_p = -1.)
    p_init = [m.mp.p; init_p * (rand() + 0.5)]
    mp = remake(m.mp, p = p_init)
    Dists = (Abs{0}(), Abs{0}(), NegAbs{0}())
    model = Modelα(mp,Dists)
    return model
end

function initialize_Modelμ(m::ModelLog, init_p = 1e-2)
    p_init = [m.mp.p; init_p * (rand() + 0.5)]
    mp = remake(m.mp, p = p_init)
    Dists = (Abs{0}(), Abs{0}(), Abs{0}())
    model = Modelμ(mp,Dists)
    return model
end

function initialize_Modelδ(m::ModelLog, init_p = 1e-2)
    p_init = [m.mp.p; init_p * (rand() + 0.5)]
    mp = remake(m.mp, p = p_init)
    Dists = (Abs{0}(), Abs{0}(), Abs{0}())
    model = Modelδ(mp,Dists)
    return model
end

import Econobio.name
function name(ma::Modelα)
    if typeof(ma.mp.st.bs[3]) <: Abs
        return "Modelαp"
    elseif typeof(ma.mp.st.bs[3]) <: NegAbs
        return "Modelαn"
    else
        return "Unknown"
    end
end