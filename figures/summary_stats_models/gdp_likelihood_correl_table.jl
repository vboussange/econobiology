#=
Exploring relationship between GDP and model likelihood.
- fits a linear regression model of likelihood as a function of GDP, and then print the coefficients of the linear regression model.
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

dfg = groupby(df_arr_sorted, ["country"])

dfg_log = [] # countries where log models are supported
dfg_eco_evo = [] # countries where eco evo models are supported
for df in dfg
    if df[df.Model .== model_nul,:ΔBIC][] < 10
        push!(dfg_log,df)
    else
        push!(dfg_eco_evo,df)
    end
end
df_log = vcat([df[df.Model .== model_nul,:] for df in dfg_log]...)
df_eco_evo = vcat([df[df.ΔBIC .== 0.,:] for df in dfg_eco_evo]...)
df = vcat(df_eco_evo,df_log) # best models

# calculating discounted likelihood
df[!,:discounted_likelihood] = df.likelihood ./ [sum(length.(r.Result.res.pred)) for r in eachrow(df)] # discounted by the number of points
df[!,:N] = [r.Result.m.mp.N for r in eachrow(df)]# number of sectors

macro_economic_indic = :gdp_last
dropmissing!(df,macro_economic_indic); sort!(df, macro_economic_indic)
x = df[:,macro_economic_indic]; 
y = df.likelihood # not discounted

# calculating stats GLM and exporting to tables
using RegressionTables
using GLM

# removing outlier (LUX)
filter!(r -> !((r.gdp_last > 1e5) && (r.likelihood < 500)), df)

# new analysis
df[!,:nyears] = fill(0., size(df,1))
[r.nyears = size(hcat(r.Result.res.pred...),2) for r in eachrow(df)]
df[!,:ndatapoints] = df.nyears .* df.N
df[!,:log_gdp] = log.(df.gdp_last)

# normalising dataframes columns
for lab in [:nyears, :ndatapoints, :likelihood, :log_gdp]
    df[!,lab] = (df[:,lab] .- mean(df[:,lab]) )./ std(df[:,lab])
end

ols_datapoints = lm(@formula(likelihood ~ ndatapoints), df)
df[!,:likelihood_residuals] = residuals(ols_datapoints)

# normalising dataframes columns
for lab in [:likelihood_residuals]
    df[!,lab] = (df[:,lab] .- mean(df[:,lab]) )./ std(df[:,lab])
end

ols_residuals = lm(@formula(likelihood_residuals ~ N + log_gdp), df)

# printing t coefficients
open("ols_gdp_likelihood.txt", "w") do io
    println(ols_datapoints)
    println(io, r2(ols_datapoints))
    println(io, ols_residuals)
    println(io, r2(ols_residuals))
end

# printing β coefficients
regtable([ols_datapoints,ols_residuals]...; renderSettings = latexOutput("gdp_likelihood_regtable.txt"),
            print_estimator_section=false,
            regression_statistics=[:nobs,:r2],
            labels = Dict("likelihood" => "Log-likelihood",
                        "likelihood_residuals" => "Log-likelihood residuals",
                        "log_gdp" => "log(GDP)",
                        "ndatapoints" => "NT",
                        "discounted_likelihood" => "Discounted log-likelihood",
                        "__LABEL_STATISTIC_N__" => "Number of countries"),
            number_regressions = false)

#=
R2 analysis

=#
ols_r2 = lm(@formula(R2 ~ N + log_gdp), df)

# printing t coefficients
open("ols_gdp_r2.txt", "w") do io
    println(io, ols_r2)
    println(io, r2(ols_r2))
end

# printing β coefficients
regtable(ols_r2; renderSettings = latexOutput("gdp_r2_regtable.txt"),
            print_estimator_section=false,
            regression_statistics=[:nobs,:r2],
            labels = Dict("likelihood" => "Log-likelihood",
                        "likelihood_residuals" => "Log-likelihood residuals",
                        "log_gdp" => "log(GDP)",
                        "ndatapoints" => "NT",
                        "discounted_likelihood" => "Discounted log-likelihood",
                        "__LABEL_STATISTIC_N__" => "Number of countries"),
                        number_regressions = false)