#=
Plotting Fig. 3.
=#

cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, PyCall, Printf
using DataFrames
using MiniBatchInference, Econobio
using Glob
using UnPack
using Statistics
include("../format.jl")
xlabs = [L"\alpha^-",L"\alpha^+", L"\mu", L"\delta"] 

scenario = "1digit"
group_size_init = 21
sigma_synthetic = 0.3

df_alphan = load("../../code/controlled_experiment/results/2022-08-11/synthetic_data_v2_alphan_1digit_without_services.jld2", "df_results")
df_alphap = load("../../code/controlled_experiment/results/2022-08-11/synthetic_data_v2_alphap_1digit_DEU_without_services.jld2", "df_results")
df_delta = load("../../code/controlled_experiment/results/2022-08-11/synthetic_data_v2_delta_1digit_without_services.jld2", "df_results")
df_mu = load("../../code/controlled_experiment/results/2022-08-11/synthetic_data_v2_mu_1digit_without_services.jld2", "df_results")
##########################
### post processing    ###
##########################
df_arr = [df_alphan, df_alphap, df_mu, df_delta]
explored_params = ["alpha", "alpha", "mu", "delta"]
df_model_arrays = []

# filtering results that did not succeed
[filter!(r -> !isnothing(r.Result), df) for df in df_arr]

for (i,df_results) in enumerate(df_arr)
    explored_param = explored_params[i]
    df_results[!,"loss"] = [r.Result.res.minloss for r in eachrow(df_results)]
    size_init = size(df_results,1)
    filter!(row ->!isnothing(row.Result), df_results)
    println("Remaining data after filter: $(size(df_results,1))/$(size_init)")

    # Keeping only best results
    # explored_param = "alpha"
    dfg = groupby(df_results, ["group_size_init","sigma_synthetic",explored_param])
    df_arr_sorted = empty(df_results) # containing only best results for each model and group size
    df_arr_sorted.std_loss = fill(NaN, size(df_arr_sorted,1))
    for df in dfg
        _dfg = groupby(df, :Model)
        for _df in _dfg
            _df[!,"std_loss"] = fill(std(_df.loss) / minimum(_df.loss),size(_df,1))
            push!(df_arr_sorted,_df[argmin(_df.loss),:])
        end
    end

    # calculating AIC, ΔAIC
    df_arr_sorted[!,"ΔBIC"] = fill(NaN,size(df_arr_sorted,1))
    df_arr_sorted[!,"BIC"] = fill(NaN,size(df_arr_sorted,1))

    # explored_param = "μ"
    dfg = groupby(df_arr_sorted, ["group_size_init","sigma_synthetic",explored_param])
    for _df in dfg
        # Recalculating AIC if needed
        for r in eachrow(_df)
            σ = estimate_σ(r.Result, r.odeadata_synthetic; noisedistrib=LogNormal(), include_ic=true)
            npoints = length(vcat(r.Result.res.ranges...))
            m = r.Result.m.mp.N * npoints
            r.BIC = m * log(σ^2) + length(r.Result.m) * log(m)
            r.AIC = m * log(σ^2) + 2 * length(r.Result.m)
        end
        _df.ΔAIC .= _df.AIC .- minimum(_df.AIC)
        _df.ΔBIC .= _df.BIC .- minimum(_df.BIC)
        println(_df[:,["Model", explored_param, "loss","likelihood", "sigma_synthetic", "ΔAIC", "ΔBIC", "group_size_init", "std_loss"]])
    end
    # replacing model names with nice names
    df_arr_sorted.Model = replace(df_arr_sorted.Model, "ModelLog" => L"\mathcal{M}_{null}",
                                                "Modelαp" => L"\mathcal{M}_{\alpha^+}",
                                                "Modelαn" => L"\mathcal{M}_{\alpha^-}",
                                                "Modelμ" => L"\mathcal{M}_{\mu}",
                                                "Modelδ" => L"\mathcal{M}_{\delta}")
    # selecting only group size of interest
    filter!(r -> r.group_size_init == group_size_init, df_arr_sorted)
    filter!(r -> r.sigma_synthetic == sigma_synthetic, df_arr_sorted)

    df_model_array = groupby(df_arr_sorted,"Model", sort=true) |> collect
    sort!.(df_model_array, Ref([explored_param, "sigma_synthetic"]))
    push!(df_model_arrays, df_model_array)
end

##########################
### technical plotting ###
##########################

fig, axs = plt.subplots(2, 4, sharey="row", sharex="col", figsize = (5.2,5), gridspec_kw = Dict("height_ratios" => [1,1.]))
fig.set_facecolor("None")
[ax.set_facecolor("None") for ax in axs]

# Plotting alpha
to_ploty = :ΔBIC
to_plotx = explored_params[1]
hyp_lab = [df.Model[1] for df in df_model_arrays[1]]
handles = []
for (i,df_model_array) in enumerate(df_model_arrays[1:2])
    for (m,_df) in enumerate(df_model_array)
        # axs[1,1].set_title(L"\sigma = %$(_df.sigma_synthetic[1])")
        if i == 1
            hdl = axs[1,i].scatter(_df[:,to_plotx], _df[:,to_ploty], color  = COLORMODELS[m], label = hyp_lab[m], marker = MARKERSTYLESMODELS[m] )
            push!(handles, hdl)
        else
            axs[1,i].scatter(_df[:,to_plotx], 
                            _df[:,to_ploty], 
                            color = COLORMODELS[m], 
                            marker = MARKERSTYLESMODELS[m]  )
        end
        axs[2,i].scatter(_df[:,to_plotx], 
                        _df[:,to_ploty], 
                        color  = COLORMODELS[m], 
                        marker = MARKERSTYLESMODELS[m] )
    end
end
[axs[2,i].set_xlabel(xlabs[i]) for i in 1:2]
axs[2,1].set_xticks([-10,-5,0])
axs[2,2].set_xticks([0,15,30])

display(fig)

# Plotting mu
to_plotx = explored_params[3]
hyp_lab = [df.Model[1] for df in df_model_arrays[3]]
for (m,_df) in enumerate(df_model_arrays[3])
        # axs[1,2].set_title(L"\sigma = %$(_df.sigma_synthetic[1])")

        axs[1,3].scatter(_df[:,to_plotx], 
                        _df[:,to_ploty], 
                        color = COLORMODELS[m], 
                        marker = MARKERSTYLESMODELS[m]  )

        axs[2,3].scatter(_df[:,to_plotx], 
                        _df[:,to_ploty], 
                        color  = COLORMODELS[m], 
                        marker = MARKERSTYLESMODELS[m] )

        axs[2,3].set_xscale("log")
end
axs[2,3].set_xlabel(xlabs[3])
display(fig)

# Plotting delta
to_plotx = explored_params[4]
hyp_lab = [df.Model[1] for df in df_model_arrays[1]]
for (m,_df) in enumerate(df_model_arrays[4])
        # axs[1,3].set_title(L"\sigma = %$(_df.sigma_synthetic[1])")

        axs[1,4].scatter(_df[:,to_plotx], 
                        _df[:,to_ploty], 
                        color = COLORMODELS[m], 
                        marker = MARKERSTYLESMODELS[m]  )

        axs[2,4].scatter(_df[:,to_plotx], 
                        _df[:,to_ploty], 
                        color  = COLORMODELS[m], 
                        marker = MARKERSTYLESMODELS[m] )

        axs[2,4].set_xscale("log")
end
axs[2,4].set_xlabel(xlabs[4])
display(fig)


# subfigs[2].supylabel(L"\Delta"*"AIC",fontsize=15)
fig.text(0.03, 0.5, L"\Delta"*"BIC", va="center", rotation="vertical", fontsize = 10)
display(fig)

fig.subplots_adjust(hspace=0.1, wspace = 0.1)
fig.legend(handles = handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.1),)

# filling areas for support
# ωspan = [-3, 3]
ylim = axs[1].get_ylim()
# s_marker = 10
for ax in axs
    xlim = ax.get_xlim()
    ax.fill_between(xlim, -0.5, 10, facecolor = "tab:green", alpha = 0.1, zorder = 0)
    # ax.fill_between(xlim, 2, 10, facecolor = "tab:grey", alpha = 0.3, zorder = 0)
    ax.fill_between(xlim, 10, ylim[2], facecolor = "tab:red", alpha = 0.1, zorder = 0)
    ax.set_xlim(xlim...)
end
axs[2].set_ylim(-0.5,10)
axs[1].set_ylim(10,ylim[2])
axs[1].set_yscale("log")

xs = -10
axs[2].annotate("Support",[xs, 8.5])
axs[1].annotate("No support",[xs, 100.])
gcf()


_let = ["A","B","C","D"]
for (i,ax) in enumerate([axs[1,1],axs[1,2], axs[1,3],axs[1,4]])
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

display(fig)

fig.savefig("synthetic_test_all_AIC_r=$sigma_synthetic.png", dpi = 300, bbox_inches="tight")