#=
 Writes the mean, median and standard deviation of the R2 values for all the best model and model results.
=#
cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, PyCall, Printf
using DataFrames
using MiniBatchInference, Econobio
using Polynomials
using Glob
using UnPack
using HypothesisTests
include("../format.jl")

scenario_name = "2022-08-28_fit_countries_all_without_services_1digit_ADAM_1e-3_df_arr_sorted"
@load "../cleaning_results/$scenario_name.jld2" df_arr_sorted


model_nul = L"$\mathcal{M}_{null}$"
group_size_init = 21
filter!(row -> (row.group_size_init) == group_size_init, df_arr_sorted)

open("r2_all.text", "w") do io
    println(io, "Mean R2 :", mean(df_arr_sorted.R2))
    println(io, "Median R2 :", mean(df_arr_sorted.R2))
    println(io, "Std R2 :", mean(df_arr_sorted.R2))
end

dfg = groupby(df_arr_sorted,:country)

dfg_log = [] # countries where log models are supported
dfg_eco_evo = [] # countries where eco evo models are supported
for df in dfg
    if df[df.Model .== L"$\mathcal{M}_{null}$",:ΔBIC][] < 10
        push!(dfg_log,df)
    else
        push!(dfg_eco_evo,df)
    end
end
println("Log model accepted ", length(dfg_log), " times")

# Calculating best model array
df_log = vcat([df[df.Model .== model_nul,:] for df in dfg_log]...)
df_eco_evo = vcat([df[df.ΔBIC .== 0.,:] for df in dfg_eco_evo]...)
df_best_model = vcat(df_eco_evo,df_log) # best models

open("r2_best_models.txt", "w") do io
    filter!(r ->!isnan(r.R2), df_best_model)
    println(io, "nb countries:", size(df_best_model,1))
    println(io, "Mean R2 :", mean(df_best_model.R2))
    println(io, "Median R2 :", median(df_best_model.R2))
    println(io, "Std R2 :", std(df_best_model.R2))
end
