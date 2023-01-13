#=
Running optimization on all available models for a particular country
* Version 
- v2:
    - NegAbs
    - initialisation with functions
    - multiple runs
=#
using Interpolations
using MiniBatchInference, OrdinaryDiffEq, DiffEqSensitivity, Econobio
using LightGraphs, SimpleWeightedGraphs
using DataFrames, JLD2, UnPack, Dates
using LinearAlgebra
using Bijectors: Identity, inverse
# using PyPlot
using LaTeXStrings
using Optimization, OptimizationOptimJL, OptimizationFlux, LineSearches
using ProgressMeter
using Distributions: LogNormal
include("../../comparision_model/1digit/inference_real_country.jl")
using Random;
if !isempty(ARGS)
    Random.seed!(parse(Int64,ARGS[1]));
end
cd(@__DIR__)

dig = "1digit"
country = "DEU"
name_scenario = "synthetic_data_v2_alphap_$(dig)_$(country)"
loop = true
verbose = true
plotting  = false
info_per_its = 100
_today = today()
nruns = 5 # /!\ if it is set to more than 1, new code should be implemented for the analysis
group_sizes = [11, 16, 21]
σ_synthetics = 0.:0.1:0.3
alphas = range(0., 40., length = 5)

# loading coefficients
# @load "../../comparision_model/1digit/results/2022-08-05/test_country1digit_USA.jld2" df_arr_sorted
@load "../../../figures/cleaning_results/2022-08-09_fit_countries_all_without_services_1digit_df_arr_sorted.jld2" df_arr_sorted
res_fitted = df_arr_sorted[(df_arr_sorted.Model .== Ref(L"\mathcal{M}_{\alpha^+}")) .* (df_arr_sorted.group_size_init .==  16) .* (df_arr_sorted.country .==  "DEU"), :Result][1]

# importing odedata, to get tsteps, tspan etc...
tsteps, N, _scale, odedata, sitc_labels, sitc_fulllabels, uworld, Σ = get_data_country(country,"1digit", todiscard= ["Unspecified","Services"])
# keeping only time steps where data is available for all products
temp_odedata = replace(x -> x < 1e-7 ? Inf : x, odedata)
minodedata = minimum(temp_odedata, dims=2)
idx_start = maximum([findfirst(temp_odedata[i,:] .== minodedata[i]) for i in 1:N])
tsteps = tsteps[idx_start:end] # services available only up to 2020
odedata = odedata[:,idx_start:end] .|> Float64
padding_data = odedata .> 1e-7;

# generating a full model with r, b coefficient from `res_fitted`
mp = remake(res_fitted.m.mp, p = [res_fitted.res.p_trained..., 0., 0., 0.], u0 = res_fitted.res.u0s_trained[1])
# mp = remake(res_fitted.m.mp, p = [(rand(N) .+ 0.5) * 0.02; (rand(N) .+ 0.5) * 5.; 0.; 0; 0.], u0 = res_fitted.res.u0s_trained[1])
synthetic_model = Modelαμδ(mp, (Abs{0}(),Abs{0}(),Identity{0}(),Abs{0}(),Abs{0}()))

#=
META PARAMETERS LEARNING
=#
sensealg = DiffEqSensitivity.ForwardDiffSensitivity()
optims = [ADAM(1e-3), BFGS(linesearch=LineSearches.BackTracking())]

alg = BS3()
continuity_term = 0. #1 / group_size_init
prior_scaling = 0. #3 / group_size_init
epochs = [1200, 800]

init_models = [initialize_ModelLog, initialize_Modelαp, initialize_Modelαn, initialize_Modelμ, initialize_Modelδ]

# initialising df and pars
iter = Iterators.product(1:length(group_sizes), 1:length(σ_synthetics), 1:length(alphas)) |> collect # required for atomic access
progr = Progress(length(iter) * nruns * (length(init_models) + 1), showspeed = true, barlen = 10)
df_results = DataFrame("Model" => String[], 
                        "Result" => Any[], 
                        "likelihood" => Float64[], 
                        "AIC" => Float64[], 
                        "ΔAIC" => Float64[],
                        "alpha" => Float64[],
                        "sigma_synthetic" => Float64[],
                        "odeadata_synthetic" => Array{Float64}[],
                        "group_size_init" => Int64[])
pars_arr = Dict{Symbol,Any}[]

if !loop # for debugging
    group_size_init = 16
    σ_synthetic = 0.2
    alpha = 40

    raw_data = simulate(synthetic_model, p = [synthetic_model.mp.p[1:2*N]; alpha; 0.; 0.])
    if plotting
        fig, ax = subplots()
        ax.plot(raw_data', linestyle = ":")
        ax.set_yscale("log")
        display(fig)
    end
    
    group_size_init = min(group_size_init, size(odedata,2))
    ic_term = 1. / group_size_init

    ranges = group_ranges(length(tsteps), group_size_init)
    tspan = (tsteps[1], tsteps[end]).|> Float64

    odedata = raw_data .* exp.(σ_synthetic .* randn(size(raw_data)...))
    parlog = Dict{Symbol,Any}(); 
    @pack! parlog = country, ranges, odedata, padding_data, optims, ic_term, prior_scaling, continuity_term, group_size_init
    # Running all other models
    for (i,m) in enumerate(init_models)
        pars = copy(parlog)
        model = initialize_ModelLog(odedata, ranges, tspan, tsteps, uworld)
        pars[:model] = m(model);
        rsim = simu(pars)
        push!(df_results,[rsim[1:end-1]...,  alpha, σ_synthetic, odedata, group_size_init])
    end
    df_results[!,"alpha"] = [getα(r.Result.m.mp.p, r.Result.m) for r in eachrow(df_results)]
    df_results[!,"loss"] = [r.Result.res.minloss for r in eachrow(df_results)]
    println(df_results[:,["Model","loss","likelihood", "alpha"]])
else
    @info "Parameter intialisation"
    for group_size_init in group_sizes, σ_synthetic in σ_synthetics, alpha in alphas
        try
            raw_data = simulate(synthetic_model, p = [synthetic_model.mp.p[1:2*N]; alpha; 0.; 0.])
            ic_term = 1. / group_size_init

            ranges = group_ranges(length(tsteps), group_size_init)
            tspan = (tsteps[1], tsteps[end]).|> Float64

            odedata = raw_data .* exp.(σ_synthetic .* randn(size(raw_data)...))
            group_size_init = min(group_size_init, size(odedata,2))

            if plotting
                fig, ax = subplots()
                ax.plot(odedata', linestyle = ":")
                ax.set_yscale("log")
                display(fig)
            end
            for _ in 1:nruns
                # Initialising log model
                parlog = Dict{Symbol,Any}(); 
                @pack! parlog = country, ranges, odedata, padding_data, optims, ic_term, prior_scaling, continuity_term, group_size_init, alpha, σ_synthetic, group_size_init
                for (i,m) in enumerate(init_models)
                    pars = copy(parlog)
                    model_init = initialize_ModelLog(odedata, ranges, tspan, tsteps, uworld)
                    pars[:model] = m(model_init); 
                    push!(pars_arr, pars)
                    push!(df_results, ("", nothing, NaN, NaN, NaN, NaN, NaN, [], 0))
                end
            end
        catch e
            println("Error with initialisation for $country", e)
            println("Skipping $country")
            continue
        end
    end
    @info "Starting loop over other models"
    Threads.@threads for k in 1:length(pars_arr)
        p = pars_arr[k]
        if !isempty(p)
            try
                @unpack alpha, σ_synthetic, odedata, group_size_init = p
                res = simu(p)
                df_results[k,:] = [res[1:end-1]..., alpha, σ_synthetic, odedata, group_size_init]
            catch e
                println("problem with model : $(p[:model]), iteration $k / $(length(pars_arr))")
                println(e)
            end
            next!(progr)
        end
    end
    dict_simul = Dict{String, Any}()
    @pack! dict_simul = df_results
    save(joinpath("results",string(_today), "$name_scenario.jld2"), dict_simul)
    println(df_results)
    println("Results saved")
end

#=
Notes :

For detecting alpha neg, we need to have some sort of growth in order
for the log model to converge more or less and provide positive growth rates.

What happens is that with the inferred growth rates of alpha positive and log model,
the growth rates are very small, and therefore the negative alpha is not detected.

On the other hand, when choosing growth coefficients with higher values,
(e.g. Modelδ with USA), then choosing an alpha positive leads to an explosion.

Looking at the USA coefficients, we find that some of the inferred coefficient seem unrealistic
(e.g. b = 1e-5) --> this is what causes explosion in finite time

what about we generate our own coefficients:
r ∈ [0, 0.2]
b ∈ [0, 5]
--> could do that, or we keep
=#