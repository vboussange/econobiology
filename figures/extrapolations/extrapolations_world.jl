#=
This code generates projections of world exports, based on fits.
Those are used in scripts located in `figure_countries`, to constructe the function
`uworld(t)`.
=#

cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, PyCall, Printf
using DataFrames
using MiniBatchInference, Econobio
using Glob
using UnPack
using SciMLBase # for remake
include("../format.jl")

scenario_name = "2022-08-28_fit_countries_all_without_services_1digit_ADAM_1e-3_df_arr_sorted"
@load "../cleaning_results/$scenario_name.jld2" df_arr_sorted

group_size_init = 21
filter!(row -> row.group_size_init == group_size_init, df_arr_sorted)

model_labs = [L"\mathcal{M}_{null}", L"\mathcal{M}_{\alpha^+}", L"\mathcal{M}_{\alpha^-}", L"\mathcal{M}_{\mu}"]

######################
# getting world data #
#####################
# function get_world()
country = "WLD"
# loading odedata to calculate likelihood
tsteps, N, _scale, odedata, sitc_labels, sitc_fulllabels, uworld, Σ = get_data_country(country,"1digit", todiscard= ["Unspecified","Services"])
# keeping only time steps where data is available for all products
temp_odedata = replace(x -> x < 1e-7 ? Inf : x, odedata)
if any(isinf.(temp_odedata))
    idx_start_all = [findlast(isinf, temp_odedata[i,:]) for i in 1:N]
    idx_start_all = idx_start_all[.!isnothing.(idx_start_all)]
    idx_start = maximum(idx_start_all)
else
    idx_start = 1
end
odedata .*= _scale

tsteps = tsteps[idx_start:end] # services available only up to 2020
odedata = odedata[:,idx_start:end] .|> Float64
padding_data = odedata .> 1e-7;

df_country = filter(row -> (row.country == country), df_arr_sorted)
# end

## DEFINING FIGURE
fig, axs = plt.subplots(2, 2, 
                        # sharey="row", 
                        # sharex=true, 
                        figsize = (5.2,5), 
                        # gridspec_kw = Dict("height_ratios" => [1,1.5])
                        )
fig.set_facecolor("None")
[ax.set_facecolor("None") for ax in axs]
colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]
linestyles = ["-", "--", ":", "-.", (0, (5,10))]
_cmap = PyPlot.cm.get_cmap("tab20", min(N,20))
color_palette = [_cmap(i-1) for i in 1:min(N,20)];

for (mi, mod) in enumerate(model_labs)
    ax = axs[mi]
    r = df_country[df_country.Model .== mod, :][1,:]
    m = r.Result.m
    N = m.mp.N
    # rscaling
    p_rescaled = [m.mp.p[1:N];m.mp.p[N+1:end]./_scale]
    mp = remake(m.mp, p = p_rescaled, u0 = r.Result.res.u0s_trained[1] * _scale)
    m = typeof(m)(mp)
    # simulations for fitted
    fitted = simulate(m)
    # simulations for predictions
    pred_horizon = 50
    new_tspan = (m.mp.tspan[1], m.mp.tspan[2] + pred_horizon)
    new_tsteps = new_tspan[1]:1:new_tspan[2]
    new_mp = remake(m.mp, tspan = new_tspan, kwargs_sol = Dict(:saveat => new_tsteps))
    new_m = typeof(m)(new_mp)
    pred = simulate(new_m)
    # plotting
    prod_to_plot = sortperm(sum(odedata,dims=2)[:], rev=true)[1:min(N,20)] #plotting only 20 biggest sectors
    for (i,p) in enumerate(prod_to_plot)
        ax.plot(tsteps, fitted[p,:], color = color_palette[i])
        ax.plot(new_tsteps[end-pred_horizon+1:end], pred[p, end-pred_horizon+1:end], color = color_palette[i], linestyle = "--")

        ax.scatter(tsteps, 
                    odedata[p,:], 
                    label = (mi==1) ? sitc_fulllabels[p] : nothing, 
                    s=5., 
                    color = color_palette[i])
    end
    ax.set_title(r.Model)
    display(fig)

    ax.set_title(mod)
    # PLOTTING AIC
    if mi == 1
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
        ylim = ax.get_ylim()
        ax.set_ylim(1e-5,ylim[2])
    end
    years = new_tsteps .+ 1962
    ax.set_xticks(new_tsteps[1:10:end])
    ax.set_xticklabels(Int.(years)[1:10:end], rotation=45)
    ax.set_yscale("log")
end
fig.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.2),)
fig.suptitle(country)
fig.tight_layout()
display(fig)
fig.savefig("World.png", dpi = 300, bbox_inches="tight")


# building uworld(t)
r = df_country[df_country.Model .== model_labs[1], :][1,:]

m = r.Result.m
N = m.mp.N
# rescaling
p_rescaled = [m.mp.p[1:N];m.mp.p[N+1:end]./_scale]
mp = remake(m.mp, p = p_rescaled, u0 = r.Result.res.u0s_trained[1] * _scale)
m = typeof(m)(mp)
# simulations for predictions
pred_horizon = 50
new_tspan = (m.mp.tspan[1], m.mp.tspan[2] + pred_horizon)
new_tsteps = new_tspan[1]:1:new_tspan[2]
new_mp = remake(m.mp, tspan = new_tspan, kwargs_sol = Dict())
new_m = typeof(m)(new_mp)
pred = simulate(new_m)
@save "uworld_pred.jld2" pred new_tsteps new_tspan sitc_labels