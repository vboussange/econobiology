#=
Plotting Fig S2 to S5, using initial conditions for each of the fitted segments
simulating nor more than for each segment.
=#

cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, PyCall, Printf
using DataFrames
using MiniBatchInference, Econobio
using Glob
using UnPack
include("../../format.jl")

# scenario_name = "synthetic_data_bis_1digit_USACHNJPNDEUBRAFRARUSGBRAUSCANINDMEXKORESPIDN"
scenario_name = "fit_countries_all_without_services_1digit_ADAM_1e-3_df_arr_sorted"
date = "2022-08-15"
# country = "AUS"
@load "../../cleaning_results/$(date)_$(scenario_name).jld2" df_arr_sorted
group_size_init = 21
filter!(row -> row.group_size_init == group_size_init, df_arr_sorted)

fignames = ["alphan", "alphap", "mu", "delta"]
countries_to_test = [["ARG", "HRV",],["HUN", "DEU",], ["POL", "FIN",], ["GBR", "BEL"]]
models_to_test = [L"\mathcal{M}_{\alpha^-}", L"\mathcal{M}_{\alpha^+}", L"\mathcal{M}_{\mu}", L"\mathcal{M}_{\delta}"]

for (sc,mod_to_test) in enumerate(models_to_test)
    countries = countries_to_test[sc]
    figname = fignames[sc]



    model_labs = [mod_to_test,L"\mathcal{M}_{null}"]
    linestyles = LINESTYLES[[1,1]]

    ## DEFINING FIGURE
    fig, axs = plt.subplots(2, 2, 
                            sharey="row", 
                            # sharex="col", 
                            figsize = (5.2,5), 
                            # gridspec_kw = Dict("height_ratios" => [1,1.5])
                            )
    fig.set_facecolor("None")
    [ax.set_facecolor("None") for ax in axs]
    _cmap = PyPlot.cm.get_cmap("tab20c", 20) # plotting only nine countrues
    color_palette = [_cmap(i-1) for i in 2:2:20];

    for (c,country) in enumerate(countries)
        println(country)
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
        
        tsteps = tsteps[idx_start:end] # services available only up to 2020
        odedata = odedata[:,idx_start:end] .|> Float64
        padding_data = odedata .> 1e-7;
        
        df_country = filter(row -> (row.country == country), df_arr_sorted)

        for (mi, mod) in enumerate(model_labs)
            ax = axs[c,mi]
            r = df_country[df_country.Model .== mod, :][1,:]
            res = r.Result.res

            for p in 1:N
                for (ri,rg) in enumerate(res.ranges)
                    # @show tsteps[rg]
                    # @show res.ranges[ri][p,:]
                    ax.plot(tsteps[rg], 
                            res.pred[ri][p,:], 
                            # label = (i == 1) && (c == 1) && (ri ==1) ? r.Model : nothing,
                            label = (mi==1) && (c == 1) && (ri ==1) ? sitc_fulllabels[p] : nothing, 
                            color = color_palette[p], 
                            linestyle = linestyles[mi],
                            linewidth=1.)
                    # if mi == 1
                    ax.scatter(tsteps[rg], 
                                odedata[p,rg], 
                                # label = (mi==1) && (c == 1) && (ri ==1) ? sitc_fulllabels[p] : nothing, 
                                s=1., 
                                color = color_palette[p],
                                # marker = "x",
                                # linewidths=0.3
                                )
                    # end
                end

            end
            
            if mi == 1
                fig.text(0.95, 1.05, country, transform=ax.transAxes, fontweight="bold")
            end

            odedata_minibatch = hcat([odedata[:,rg] for rg in res.ranges]...)
            fitted = hcat(res.pred...)
            ax.text(0.1,0.9,string(L"R^2 = ", @sprintf "%2.3f" R2(odedata_minibatch, fitted, LogNormal())), transform = ax.transAxes)    

            display(fig)
        
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
            end
            c == 1 ? ax.set_title(mod) : nothing
            years = tsteps .+ 1962
            ax.set_xticks(tsteps[1:6:end])
            ax.set_xticklabels(Int.(years)[1:6:end], rotation=45)
            ax.set_yscale("log")
            mi == 1 ? ax.set_ylabel("Scaled economic sector\ncapital, "*L"n^{(c)}_i") : nothing
        end
    end
    for ax in axs
        ylim = ax.get_ylim()
        ax.set_ylim(1e-5,ylim[2])
    end
    fig.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.1),)

    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.35)
    display(fig)
    fig.savefig("figure_$(figname)_minibatch.png", dpi = 300, bbox_inches="tight")
end
