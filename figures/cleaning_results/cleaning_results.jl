#=
Post processing and cleaning results.
=#

cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, PyCall, Printf
using DataFrames
using MiniBatchInference, Econobio
using Glob
using UnPack
include("../format.jl")

# scenario_name = "synthetic_data_bis_1digit_USACHNJPNDEUBRAFRARUSGBRAUSCANINDMEXKORESPIDN"
scenario_name = "fit_countries_all_without_services_1digit_ADAM_1e-3"
date = "2022-08-28"
# country = "AUS"
@load "../../code/inference_1digit/results/$(date)/$(scenario_name).jld2" df_results
size_init = size(df_results,1)
filter!(row ->!isnothing(row.Result), df_results)
df_results[!,"loss"] = [r.Result.res.minloss for r in eachrow(df_results)]
df_results[!,"wa"] = fill(NaN,size(df_results,1))
df_results[!,"waBIC"] = fill(NaN,size(df_results,1))
df_results[!,"scale"] = fill(NaN,size(df_results,1))
# re calculating AIC, ΔAIC
df_results[!,"ΔBIC"] = fill(NaN,size(df_results,1))
df_results[!,"BIC"] = fill(NaN,size(df_results,1))
df_results[!,"rel_BIC"] = fill(NaN,size(df_results,1))
df_results[!,"R2"] = fill(NaN,size(df_results,1))

println("Remaining data after filter: $(size(df_results,1))/$(size_init)")

# POST PROCESSING
# Keeping only best results
dfg = groupby(df_results, ["group_size_init","country"])
df_arr_sorted = empty(df_results) # containing only best results for each model and group size
df_arr_sorted.std_loss = fill(NaN, size(df_arr_sorted,1))
for df in dfg
    _dfg = groupby(df, :Model)
    for _df in _dfg
        _df[!,"std_loss"] = fill(std(_df.loss) / minimum(_df.loss),size(_df,1))
        push!(df_arr_sorted,_df[argmin(_df.loss),:])
    end
end

dfg = groupby(df_arr_sorted, ["group_size_init","country"])
for _df in dfg
    country = _df.country[1]
    println(country)
    if occursin("without", scenario_name) # keeping end point
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
    else
        tsteps, N, _scale, odedata, sitc_labels, sitc_fulllabels, uworld, Σ = get_data_country(country,"1digit", todiscard= ["Unspecified"])
        # keeping only time steps where data is available for all products
        temp_odedata = replace(x -> iszero(x) ? Inf : x, odedata)
        minodedata = minimum(temp_odedata, dims=2)
        idx_start = maximum([findfirst(temp_odedata[i,:] .== minodedata[i]) for i in 1:N])
        tsteps = tsteps[idx_start:end-1] # services available only up to 2020
        odedata = odedata[:,idx_start:end-1] .|> Float64
        padding_data = odedata .> 1e-5;
    end

    _df.Model = replace(_df.Model, "ModelLog" => L"\mathcal{M}_{null}",
                                            "Modelαp" => L"\mathcal{M}_{\alpha^+}",
                                            "Modelαn" => L"\mathcal{M}_{\alpha^-}",
                                            "Modelμ" => L"\mathcal{M}_{\mu}",
                                            "Modelδ" => L"\mathcal{M}_{\delta}")

    sort!(_df,"Model")
    myparams = ["α", "μ", "δ"]
    methods = "get".*myparams .|> Symbol
    for i in 1:length(methods)
        _df[!,"$(myparams[i])_trained"] = [eval(methods[i])(r.Result.m.mp.p, r.Result.m) for r in eachrow(_df)]
    end
    _df[!,"scale"] = fill(_scale,size(_df,1))

    # Recalculating AIC if needed
    for r in eachrow(_df)
        σ = estimate_σ(r.Result, odedata, noisedistrib=LogNormal(), include_ic = true)
        npoints = length(vcat(r.Result.res.ranges...))
        m = r.Result.m.mp.N * npoints
        r.BIC = m * log(σ^2) + length(r.Result.m) * log(m)
        r.AIC = m * log(σ^2) + 2 * length(r.Result.m)

        # R2 calculation
        res = r.Result.res
        odedata_minibatch = hcat([odedata[:,rg] for rg in res.ranges]...)
        pred = hcat(res.pred...)
    
        r.R2 = R2(odedata_minibatch, pred, LogNormal())
    end
    _df.ΔAIC .= _df.AIC .- minimum(_df.AIC)
    _df.ΔBIC .= _df.BIC .- minimum(_df.BIC)
    _df.waBIC .= exp.(-_df.ΔBIC/2) ./ sum(exp.(-_df.ΔBIC/2))
    _df.wa .= exp.(-_df.ΔAIC/2) ./ sum(exp.(-_df.ΔAIC/2))

    lnull = _df[_df.Model .== L"$\mathcal{M}_{null}$","ΔBIC"][1]
    for r in eachrow(_df)
        # if r.Model !== L"$\mathcal{M}_{null}$" # this is calculated later on
        r.rel_BIC = r.ΔBIC - lnull
        # end
    end
    println(_df[:,["Model", "loss", "α_trained", "μ_trained", "δ_trained","likelihood", "ΔAIC", "ΔBIC"]])                     
end

io = open("$(date)_$(scenario_name)_df_arr_sorted.txt", "w")
dfg_g = groupby(df_arr_sorted,:group_size_init)
dfg_filtered = []
for dfg in dfg_g
    ## printing the number of countries per group_size
    dfc_g = groupby(dfg,:country)
    println(io, "Nb of countries for group_size = ",dfg[1,:group_size_init], " :", size(dfc_g) )

    ## filtering r2 negative
    for dfc in dfc_g
        r2_log = dfc[dfc.Model .== L"\mathcal{M}_{null}", :R2][]
        if r2_log < 0
            println(io, "country", dfc.country[1], " with group size ", dfc.group_size_init[1]," rejected because of R2 of log model negative")
        else
            push!(dfg_filtered, dfc)
        end
    end
end

## concatenating, for final post processed df
df_arr_sorted = vcat(dfg_filtered...)

## calculating stats per group_size
dfg_g = groupby(df_arr_sorted,:group_size_init)
for dfg in dfg_g
    if !isempty(dfg)
        println(io, "median std loss for group size init ", dfg.group_size_init[1], " : ", median(dfg.std_loss))
        println(io, "mean std loss for group size init ", dfg.group_size_init[1], " : ", mean(dfg.std_loss))
        println(io, "std std loss for group size init ", dfg.group_size_init[1], " : ", std(dfg.std_loss))
    end
end
## getting macroecnomic indicators, and saving
## this removes WLD from results,
df_arr_sorted = get_economics(df_arr_sorted)
@save "$(date)_$(scenario_name)_df_arr_sorted.jld2" df_arr_sorted
close(io)