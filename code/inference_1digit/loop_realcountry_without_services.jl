#=
Running optimization on all available models for a particular country
- Discarding services, because data only available from 1980 onwards
- Discarding "Unspecified", because considering this category as a sector 
    does not make sense, and also because its dynamics looks like fluctuating randomly.

* Version 
- v2:
    - NegAbs
    - initialisation with functions
    - multiple runs

/!\ verbose is set to false for differential equation solver, 
meaning that it will be harder to debug. Can be changed in file "inference_real_country.jl"
under modellog.mp.kwargs_sol
=#
cd(@__DIR__)
using Interpolations
using MiniBatchInference, OrdinaryDiffEq, DiffEqSensitivity, Econobio
using LightGraphs, SimpleWeightedGraphs
using DataFrames, JLD2, UnPack, Dates
using LinearAlgebra
using Bijectors: Identity, inverse
# using PyPlot
using Optimization, OptimizationOptimJL, OptimizationFlux, LineSearches
using ProgressMeter
using Distributions: LogNormal
include("inference_real_country.jl")
using Random;
if !isempty(ARGS)
    global myseed = parse(Int64,ARGS[1])
    Random.seed!(myseed);
end

dig = "1digit"
loop = true
countries = COUNTRIES_ALL #["USA" "CHN" "JPN" "DEU" "BRA" "FRA" "RUS" "GBR" "AUS" "CAN" "IND" "MEX" "KOR" "ESP" "IDN"]
name_scenario = "fit_countries_all_without_services_$(dig)_ADAM_1e-3"
verbose_loss = true
plotting  = false
info_per_its = 100
_today = today()
nruns = 5 # /!\ if it is set to more than 1, new code should be implemented for the analysis
group_sizes = [11, 16, 21, 41]

#=
META PARAMETERS LEARNING
=#
sensealg = DiffEqSensitivity.ForwardDiffSensitivity()
optims = [ADAM(1e-3), BFGS(linesearch=LineSearches.BackTracking())]
alg = BS3()
continuity_term = 0. #1 / group_size_init
prior_scaling = 0.#3 / group_size_init
epochs = [1200, 800]

init_models = [initialize_ModelLog, initialize_Modelαp, initialize_Modelαn, initialize_Modelμ, initialize_Modelδ]

# initialising df and pars
progr = Progress(length(group_sizes) * nruns * length(countries) * (length(init_models)), showspeed = true, barlen = 10)
df_results = DataFrame("Model" => String[], 
                        "Result" => Any[], 
                        "likelihood" => Float64[], 
                        "AIC" => Float64[], 
                        "ΔAIC" => Float64[],
                        "country" => String[],
                        "group_size_init" => Int64[])
pars_arr = Dict{Symbol,Any}[]
for country in countries, group_size_init in group_sizes, _ in 1:nruns
    # loading data
    println("Loading country $country")
    try
        tsteps, N, _scale, odedata, sitc_labels, sitc_fulllabels, uworld, Σ = get_data_country(country,"1digit", todiscard= ["Unspecified","Services"])
        # keeping only time steps where data is available for all products
        temp_odedata = replace(x -> x < 1e-7 ? Inf : x, odedata)
        if any(isinf.(temp_odedata))
            idx_start_all = [findlast(isinf, temp_odedata[i,:]) for i in 1:N]
            idx_start_all = idx_start_all[.!isnothing.(idx_start_all)]
            idx_start = maximum(idx_start_all) + 1
        else
            idx_start = 1
        end
        
        tsteps = tsteps[idx_start:end] # services available only up to 2020
        odedata = odedata[:,idx_start:end] .|> Float64
        padding_data = odedata .> 1e-7;

        if N > 1
            group_size_init = min(group_size_init, size(odedata,2))
            ic_term = 1 / group_size_init

            ranges = group_ranges(length(tsteps), group_size_init)
            tspan = (tsteps[1], tsteps[end]).|> Float64

            parlog = Dict{Symbol,Any}(); 
            @pack! parlog = country, ranges, odedata, padding_data, optims, ic_term, prior_scaling, continuity_term, group_size_init
            for (i,m) in enumerate(init_models)
                pars = copy(parlog)
                model_init = initialize_ModelLog(odedata, ranges, tspan, tsteps, uworld)
                pars[:model] = m(model_init); 
                push!(pars_arr, pars)
                push!(df_results, ("", nothing, NaN, NaN, NaN, "", 0))
            end
        end
    catch e
        println("Error with initialisation for $country \n", e)
        println("Skipping $country")
        continue
    end
end


@info "Starting loop over fits"
Threads.@threads for k in 1:length(pars_arr)
    p = pars_arr[k]
    if !isempty(p)
        try
            @unpack group_size_init = p
            df_results[k,:] = [simu(p)...,group_size_init]
        catch e
            println("problem with model : $(p[:model]), iteration $k / $(length(pars_arr))")
            println(e)
        end
        next!(progr)
    end
end
dict_simul = Dict{String, Any}()
@pack! dict_simul = df_results, myseed
save(joinpath("results",string(_today), "$name_scenario.jld2"), dict_simul)
println(df_results)
println("Results saved")