#=
Plotting figure 2
=#

cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions, HypothesisTests
using PyPlot, PyCall, Printf
using DataFrames
using MiniBatchInference, Econobio
using Glob
using UnPack
using Polynomials
using SciMLBase
include("../format.jl")

# scenario_name = "synthetic_data_bis_1digit_USACHNJPNDEUBRAFRARUSGBRAUSCANINDMEXKORESPIDN"
scenario_name = "fit_countries_all_without_services_1digit_ADAM_1e-3_df_arr_sorted"
date = "2022-08-28"
# country = "AUS"
@load "../cleaning_results/$(date)_$(scenario_name).jld2" df_arr_sorted
pred_world, sitc_labels_world = load("../extrapolations/uworld_pred.jld2", "pred", "sitc_labels")

group_size_init = 21
filter!(row -> row.group_size_init == group_size_init, df_arr_sorted)

model_labs = [L"\mathcal{M}_{\alpha^+}",L"\mathcal{M}_{null}"]
linestyles = LINESTYLES[[2,5]]

## DEFINING FIGURE
fig, axs = plt.subplots(2, 2, 
                        # sharey="row", 
                        # sharex="col", 
                        figsize = (5.2,5), 
                        # gridspec_kw = Dict("height_ratios" => [1,1.5])
                        )
fig.set_facecolor("None")
[ax.set_facecolor("None") for ax in axs]

for (c,country) in enumerate(["DEU", "ARE", "FRA"])
    println(country)
    # loading odedata to calculate likelihood
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
    
    df_country = filter(row -> (row.country == country), df_arr_sorted)
    ax = axs[c]

    r = df_country[df_country.ΔBIC .== 0., :][1,:]
    m = r.Result.m
    res = r.Result.res
    p_trained = res.p_trained

    # reconstructing m because of uworld
    mp_new = remake(m.mp, uworld = uworld)
    m_new = typeof(m)(mp_new)
    # simulations for fitted
    fitted = simulate(m_new, u0 = res.u0s_trained[1], p = p_trained)

    # simulations for predictions
    function new_uworld(t)
        if t <= m.mp.tspan[2]
            return uworld(t)
        else
            return pred_world(t)[sitc_labels_world .∈ Ref(sitc_labels)] ./ _scale
        end
    end
    pred_horizon = 30
    new_tspan = (m.mp.tspan[1], m.mp.tspan[2] + pred_horizon)
    new_tsteps = new_tspan[1]:1:new_tspan[2]
    new_mp = remake(m.mp, tspan = new_tspan, 
                        kwargs_sol = Dict(:saveat => new_tsteps),
                        uworld = new_uworld)
    new_m = typeof(m)(new_mp)
    pred = simulate(new_m, u0 = r.Result.res.u0s_trained[1])
    # plotting
    for i in 1:size(odedata,1)
        p = findfirst(sitc_labels[i] .== SITC_LABELS_1DIG)
        ax.plot(tsteps, 
                fitted[i,:], 
                label = (i == 1) && (c == 1) ? "Fitted" : nothing,
                color = COLOR_PALETTE_1DIG[p], 
                linewidth=0.8)
        ax.plot(new_tsteps[end-pred_horizon+1:end], 
                pred[i, end-pred_horizon+1:end], 
                label = (i == 1) && (c == 1) ? "Predicted" : nothing,
                color = COLOR_PALETTE_1DIG[p], linestyle = "--",
                linewidth=0.8)

        ax.scatter(tsteps, 
                    odedata[i,:], 
                    label = (c == 1) ? sitc_fulllabels[p] : nothing, 
                    s=3., 
                    color = COLOR_PALETTE_1DIG[p])
    end
    display(fig)
    ax.text(0.05,0.9,string(L"R^2 = ", @sprintf "%2.3f" r.R2), transform = ax.transAxes)    
    ax.set_ylabel("Scaled economic sector\ncapital, "*L"n^{(c)}_i")

    # PLOTTING AIC
    axins = mplt.inset_axes(ax, 
                            width=0.7, 
                            height=0.5,
                            bbox_to_anchor=(0.5, 0.08),
                            bbox_transform=ax.transAxes, 
                            loc=3,
                            )
    # axins.set_title(country,x=1.15, fontsize=12, fontweight="bold",)
    for (i,r) in enumerate(eachrow(df_country))
        axins.barh([i],[r.ΔBIC], 
                    color = COLORMODELS[i],
                    height = 0.3
                    )
    end
    axins.set_yticks(1:size(df_country,1))
    # axins.set_xticks([0,10,20])
    axins.set_yticklabels(df_country.Model)
    axins.set_title(L"\Delta"*"BIC")
    axins.set_facecolor("None")
    # for (i,lab) in enumerate(labels)
    #     axins.annotate(lab, (i -0.5,df_country[i,:ΔBIC]+3), fontsize=8, rotation = 30)
    # end
    display(fig)
    # ylim = axins.get_ylim()
    # axins.set_ylim(0,ylim[2]+4)

    axins.spines["right"].set_visible(false)
    axins.spines["top"].set_visible(false)

    ax.set_title(country)
    years = new_tsteps .+ 1962
    ax.set_xticks(new_tsteps[1:10:end])
    ax.set_xticklabels(Int.(years)[1:10:end], rotation=45)
    ax.set_yscale("log")
end
for ax in axs[1:3]
    ylim = ax.get_ylim()
    ax.set_ylim(1e-5,ylim[2])
end

fig.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25),)


# GDP

model_nul = L"$\mathcal{M}_{null}$"

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

macro_economic_indic = :gdp_last
dropmissing!(df,macro_economic_indic); sort!(df, macro_economic_indic)
x = df[:,macro_economic_indic]; 
y = df.likelihood # not discounted

ax = axs[4]
ax.scatter(x, y, s=5.,)
p = Polynomials.fit(log.(x),y,1)
ax.plot(x,
            p.(log.(x)),
            color = "black",
            alpha = 0.5
            )
ax.set_xlabel("GDP per capita constant USD")
ax.set_ylabel("Best model loglikelihood")
# ax.set_yscale("log")
ax.set_xscale("log")
ct = CorrelationTest(log.(x),  y)
ax.text(0.05,0.8,L"\rho =" *" $(@sprintf "%.2f" ct.r)\np = $(@sprintf "%.2e" pvalue(ct))",
            transform = ax.transAxes, fontsize = 8)

idx_countries = [1,15,20,35,40,45,50, 55, 60]
[ax.annotate(r.country, (r[macro_economic_indic],r.likelihood)) for r in eachrow(df[idx_countries,:])]

fig.tight_layout()
display(fig)

_let = ["A","B","C","D"]
for (i,ax) in enumerate(axs[[1,3,2,4]])
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
fig.savefig("figure_countries.png", dpi = 300, bbox_inches="tight")
