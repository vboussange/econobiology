#=
Running optimization on all available models for a particular country
* Version 
- v2:
    - NegAbs
    - initialisation with functions
    - multiple runs
    - only test (see inference_loop for loop over all countries)
- without services: removing services
=#
cd(@__DIR__)
using Interpolations
using MiniBatchInference, OrdinaryDiffEq, DiffEqSensitivity, Econobio
using LightGraphs, SimpleWeightedGraphs
using DataFrames, JLD2, UnPack, Dates
using LinearAlgebra
# using Random; Random.seed!(3);
using Bijectors: Identity, inverse
using PyPlot
using Optimization, OptimizationOptimJL, OptimizationFlux, LineSearches
using ProgressMeter
using Distributions: LogNormal
include("inference_real_country.jl")

dig = "1digit"
name_scenario = "test_country_$(dig)_without_services"
verbose_loss = true
plotting  = true
info_per_its = 100
_today = today()
nruns = 1 # /!\ if it is set to more than 1, new code should be implemented for the analysis

#=
META PARAMETERS LEARNING
=#
sensealg = DiffEqSensitivity.ForwardDiffSensitivity()
optims = [ADAM(1e-3), BFGS(linesearch=LineSearches.BackTracking())]
alg = BS3()

group_size_init=21
continuity_term = 0. #1 / group_size_init
prior_scaling = 0.#3 / group_size_init
epochs = [1200, 800]

init_models = [initialize_Modelαp, initialize_Modelαn, initialize_Modelμ, initialize_Modelδ]

df_results = DataFrame("Model" => String[], 
                "Result" => Any[], 
                "likelihood" => Float64[], 
                "AIC" => Float64[], 
                "ΔAIC" => Float64[],
                "country" => String[],
                "group_size_init" => Int64[])

# importing odedata
country = "CZE"
tsteps, N, _scale, odedata, sitc_labels, sitc_fulllabels, uworld, Σ = get_data_country(country,"1digit", todiscard= ["Unspecified", "Services"])
# keeping only time steps where data is available for all products
temp_odedata = replace(x -> iszero(x) ? Inf : x, odedata)
minodedata = minimum(temp_odedata, dims=2)
idx_start = maximum([findfirst(temp_odedata[i,:] .== minodedata[i]) for i in 1:N])
tsteps = tsteps[idx_start:end-1] # services available only up to 2020
odedata = odedata[:,idx_start:end-1]
padding_data = odedata .> 1e-5

# constraining the predicted unconstrained values to some range
# minlog = 5 * log.(minimum(x -> iszero(x) ? Inf : x, odedata, dims=2))
# maxlog = 0.2 * log.(maximum(x -> iszero(x) ? -Inf : x, odedata, dims=2))

# defining the time range and time span
ranges = group_ranges(length(tsteps), group_size_init)
tspan = (tsteps[1], tsteps[end]).|> Float64

# Initialising log model
model = initialize_ModelLog(odedata, ranges, tspan, tsteps, uworld)
parlog = Dict{Symbol,Any}(); 
@pack! parlog = country, ranges, model, odedata, padding_data, optims, ic_term, prior_scaling, continuity_term
@info "Starting initialisation of $model"
r_log = simu(parlog)
push!(df_results,[r_log...,group_size_init])

# Initialising all other models
res = r_log[2]
for (i,m) in enumerate(init_models)
    model = initialize_ModelLog(odedata, ranges, tspan, tsteps, uworld)
    pars = copy(parlog)
    pars[:model] = m(model); pars[:optims] = optims
    @info "Starting initialisation of $(pars[:model])"
    push!(df_results,[simu(pars)...,group_size_init])
end
df_results[!,"loss"] = [r.Result.res.minloss for r in eachrow(df_results)]
myparams = ["α", "μ", "δ"]
methods = "get".*myparams .|> Symbol
for i in 1:length(methods)
    df_results[!,"$(myparams[i])_trained"] = [eval(methods[i])(r.Result.m.mp.p, r.Result.m) for r in eachrow(df_results)]
end
println(df_results[:,["Model","loss","likelihood", "α_trained", "μ_trained", "δ_trained", "group_size_init"]])
dict_simul = Dict{String, Any}()
@pack! dict_simul = df_results
save(joinpath("results",string(_today), "$(name_scenario)_$(country).jld2"), dict_simul)

# 08-08-2020
#= 
FRA
 Row │ Model     loss      likelihood  α_trained    μ_trained      δ_trained  group_size_init 
     │ String    Float64   Float64     Float64      Float64        Float64    Int64           
─────┼────────────────────────────────────────────────────────────────────────────────────────
   1 │ ModelLog  0.174896     2024.81  NaN          NaN            NaN                     11
   2 │ Modelαp   0.174905     2024.79    0.0137721  NaN            NaN                     11
   3 │ Modelαn   0.171014     2032.89   -3.42488    NaN            NaN                     11
   4 │ Modelμ    0.180098     2020.24  NaN            9.71816e-10  NaN                     11
   5 │ Modelδ    0.16373      2044.16  NaN          NaN              0.18265               11

 Row │ Model     loss      likelihood  α_trained    μ_trained     δ_trained  group_size_init 
     │ String    Float64   Float64     Float64      Float64       Float64    Int64           
─────┼───────────────────────────────────────────────────────────────────────────────────────
   1 │ ModelLog  0.174045     1825.92  NaN          NaN           NaN                     16
   2 │ Modelαp   0.165386     1841.04    9.30298    NaN           NaN                     16
   3 │ Modelαn   0.174045     1825.92   -1.0026e-6  NaN           NaN                     16
   4 │ Modelμ    0.171101     1830.56  NaN            0.00586813  NaN                     16
   5 │ Modelδ    0.162846     1851.04  NaN          NaN             1.46577               16
 Row │ Model     loss      likelihood  α_trained     μ_trained     δ_trained  group_size_init 
     │ String    Float64   Float64     Float64       Float64       Float64    Int64           
─────┼────────────────────────────────────────────────────────────────────────────────────────
   1 │ ModelLog  0.174045     1825.92  NaN           NaN           NaN                     16
   2 │ Modelαp   0.165386     1841.04    9.30513     NaN           NaN                     16
   3 │ Modelαn   0.174045     1825.93   -3.20522e-6  NaN           NaN                     16
   4 │ Modelμ    0.171102     1830.56  NaN             0.00586648  NaN                     16
   5 │ Modelδ    0.162752     1851.32  NaN           NaN             1.46534               16

AUS
 Row │ Model     loss      likelihood  α_trained     μ_trained      δ_trained     group_size_init 
     │ String    Float64   Float64     Float64       Float64        Float64       Int64           
─────┼────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ModelLog  0.228799     1520.44  NaN           NaN            NaN                        16
   2 │ Modelαp   0.228799     1520.44    1.52322e-8  NaN            NaN                        16
   3 │ Modelαn   0.227274     1520.64   -3.33936     NaN            NaN                        16
   4 │ Modelμ    0.237942     1519.8   NaN             9.86525e-10  NaN                        16
   5 │ Modelδ    0.228799     1520.46  NaN           NaN              2.44529e-8               16

 Row │ Model     loss      likelihood  α_trained     μ_trained      δ_trained    group_size_init 
     │ String    Float64   Float64     Float64       Float64        Float64      Int64           
─────┼───────────────────────────────────────────────────────────────────────────────────────────
   1 │ ModelLog  0.228799     1520.44  NaN           NaN            NaN                       16
   2 │ Modelαp   0.237945     1519.85    2.13173e-6  NaN            NaN                       16
   3 │ Modelαn   0.227489     1519.38   -3.53944     NaN            NaN                       16
   4 │ Modelμ    0.237927     1519.8   NaN             4.89048e-10  NaN                       16
   5 │ Modelδ    0.228846     1520.43  NaN           NaN              0.0014426               16
=#