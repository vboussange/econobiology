#=
- Loads results and filters it by segment size.
- Calculates the best model for each country. 
- Plots the distribution of the best model across all countries using pie chart, 
showing the number of times each model is the best model among the countries in the dataset.
=#
cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, PyCall, Printf
using DataFrames
using MiniBatchInference, Econobio
using Glob
using UnPack
using HypothesisTests
include("../format.jl")

scenario_name = "2022-08-28_fit_countries_all_without_services_1digit_ADAM_1e-3_df_arr_sorted"
@load "../cleaning_results/$scenario_name.jld2" df_arr_sorted

model_labels = [L"$\mathcal{M}_{\alpha^+}$", L"$\mathcal{M}_{\alpha^-}$", L"$\mathcal{M}_{\delta}$", L"$\mathcal{M}_{\mu}$", L"$\mathcal{M}_{null}$"]
model_nul = model_labels[end]
group_size_init = 21
filter!(row -> row.group_size_init == group_size_init, df_arr_sorted)
# filter!(row -> group_size_init-4 <= (row.group_size_init) <= group_size_init, df_arr_sorted)

dfg = groupby(df_arr_sorted, ["group_size_init","country"])
@assert length(dfg) <= length(unique(df_arr_sorted.country))
countries = sort!([df.country[1] for df in dfg])
println("nb. of countries retained:", length(countries))
println(countries)

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

fig = plt.figure(figsize = (5.2,8))
gs = fig.add_gridspec(3,2)

ax_1 = fig.add_subplot(py"$(gs)[0,1]")
ax_2 = fig.add_subplot(py"$(gs)[1,1]")
ax_3 = fig.add_subplot(py"$(gs)[2,1]")

ax_all = fig.add_subplot(py"$(gs)[0:3,0]")
display(fig)


# Best model figure
ax = ax_1
size_models = []
for (i,modelab) in enumerate(model_labels)
    _df = filter(r -> (r.Model == modelab) , df_best_model)
    push!(size_models,size(_df,1))
end
absolute_value(val) = round(Int, val/100 * sum(size_models))
explode = [fill(0.1,length(model_labels)-1);0.05]
ax.pie(size_models, 
        labels = model_labels, 
        colors = COLORMODELS, 
        autopct=absolute_value, 
        explode = explode,
        textprops=Dict("color"=>"black", 
                        # "fontsize" => 8
                        ))

ax.set_xlabel(L"Nb. of countries where
 $\mathcal{M}_i$ is the best model")
# ax.set_xticks(1:length(model_labels))
# ax.set_xticklabels(model_labels)
display(fig)


# support against log model
ax = ax_2
size_models = []
for (i,modelab) in enumerate(model_labels[1:end-1])
    _df = filter(r -> (r.Model == modelab) && (r.rel_BIC < -10) , df_arr_sorted)
    push!(size_models,size(_df,1))
end
absolute_value(val) = round(Int, val/100 * sum(size_models))
explode = fill(0.05,length(model_labels)-1)
ax.pie(size_models, 
        labels = model_labels[1:end-1], 
        colors = COLORMODELS[1:end-1], 
        autopct=absolute_value, 
        explode = explode, 
        textprops=Dict("color"=>"black", 
                        # "fontsize" => 8
                        ))

ax.set_xlabel(L"Nb. of countries where
    $\mathcal{M}_i$ is supp. against $\mathcal{M}_{null}$")
# ax.set_xticks(1:length(model_labels))
# ax.set_xticklabels(model_labels)
display(fig)


# strength of evidence figure
ax = ax_3
for (i,mlab) in enumerate(model_labels[1:end-1])
    df = filter(r -> (r.rel_BIC < -10) && r.Model == mlab, df_arr_sorted)
    flierprops = Dict("marker"=>"o", 
                    "markerfacecolor"=>COLORMODELS[i], 
                    "markersize"=>3,
                    "linestyle"=>"none", "markeredgecolor"=>COLORMODELS[i])


    bplot = ax.boxplot([df[:,:rel_BIC]], positions = [i], 
                        patch_artist=true,  # fill with color
                        flierprops = flierprops
                        )
    for patch in bplot["boxes"]
        patch.set_facecolor(COLORMODELS[i])
        patch.set_edgecolor(COLORMODELS[i])
    end
    for item in ["caps", "whiskers","medians"]
        for patch in bplot[item]
            patch.set_color("black")
        end
    end
end

ax.set_ylabel(L"Strength of evidence of $\mathcal{M}_i$
against $\mathcal{M}_{null}$, $BIC_{\mathcal{M}_i} - BIC_{\mathcal{M}_{null}}$")
ax.set_xticks(1:length(model_labels)-1)
ax.set_xticklabels(model_labels[1:end-1])
# ax.set_yscale("log")
display(fig)

# Big summary figure
ax = ax_all
dfg = dfg_eco_evo
# sorting countries by gdp
sort_idx = sortperm([minimum(df.rel_BIC #./ sum(length.(df.Result[1].res.pred))
                            ) for df in dfg], rev=true)
dfg = dfg[sort_idx]
for (i,df) in enumerate(dfg)
    sort!(df,"Model")
    for r in eachrow(df)
            if r.Model !==  L"$\mathcal{M}_{null}$" && r.rel_BIC < -10
                npoints = sum(length.(r.Result.res.pred))
                x = r.rel_BIC #/ npoints
                j = findfirst(model_labels .== r.Model)
                ax.scatter(x, 
                            i,
                            label = r.country == "FIN" ? r.Model : nothing, # all models supported
                            marker= MARKERSTYLESMODELS[j], 
                            color = COLORMODELS[j])
            end
    end
end
ax.set_yticks(1:length(dfg))
ax.set_yticklabels([df.country[1] for df in dfg])
ax.invert_xaxis()
ax.set_xlabel("BIC"*L"_{\mathcal{M}} - "*"BIC"*L"_{\mathcal{M}_{null}}")
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.55, 1.085),)
ax.set_ylim(0,length(dfg)+1)
_let = ["A","B","C","D"]
for (i,ax) in enumerate([ax_all,ax_1,ax_2,ax_3])
    _x = -0.1
    ax.text(_x, 1.05, _let[i],
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        transform=ax.transAxes ,
        zorder = 199
    )
end


fig.tight_layout()
display(fig)
fig.savefig("figure_aggregate_countriesv2_evidence_for_eco_evo.png", dpi= 300, bbox_inches = "tight")