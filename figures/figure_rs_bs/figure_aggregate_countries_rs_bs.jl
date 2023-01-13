#=
Plotting Fig. 5
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


# calulcating the r / b ratio for each entry
df_arr_sorted[!,"ratio_rb"] = fill([], size(df_arr_sorted,1))
df_arr_sorted[!,"median_r"] = fill([], size(df_arr_sorted,1))
df_arr_sorted[!,"median_b"] = fill([], size(df_arr_sorted,1))

for r in eachrow(df_arr_sorted)
    N = r.Result.m.mp.N
    rs = r.Result.m.mp.p[1:N]
    bs = r.Result.m.mp.p[N+1:2*N]
    # r.ratio_rb = median(rs) / median(bs)
    r.median_r = rs
    r.median_b = bs / r.scale

end


model_nul = L"$\mathcal{M}_{null}$"
group_size_init = 21
filter!(row -> (row.group_size_init) == group_size_init, df_arr_sorted)

dfg = groupby(df_arr_sorted, ["country"])

# RELATIVE STRENGTH OF SUPPORT
for df in dfg
    df[!,"rel_BIC"] = fill(NaN, size(df,1))
    lnull = df[df.Model .== L"$\mathcal{M}_{null}$","ΔBIC"][1]
    for r in eachrow(df)
        if r.Model !== L"$\mathcal{M}_{null}$" # this is calculated later on
            r.rel_BIC = r.ΔBIC - lnull
        end
    end
end

# calculating relative strength of support for log model
for df in dfg
    lnull = minimum(df[.!(df.Model .== model_nul), "ΔBIC"])
    ridx = (df.Model .== model_nul)
    df[ridx,:rel_BIC] = df[ridx, "ΔBIC"] .- lnull
end

fig = figure(figsize = (5.2,2.5,),)

gs = fig.add_gridspec(2,2)

ax_1 = fig.add_subplot(py"$(gs)[0:2,0]")
ax_2 = fig.add_subplot(py"$(gs)[0,1]")
ax_3 = fig.add_subplot(py"$(gs)[1,1]", sharex =ax_2)

axs = [ax_1, ax_2, ax_3]

median_label = ["median_r", "median_b", "median_b"]
median_ylabel = [
                L"Growth coefficient, $r_i$",
                L"Self-limitation coefficient, $b_i$",
                L"Self-limitation coefficient, $b_i$"]
for (j,ax) in enumerate(axs)
    for (i,mod) in enumerate(model_labels[1:end-1])
        df = filter(r -> (r.Model == mod) && (r.rel_BIC < -10), df_arr_sorted)
        distrib = vcat(df[:,median_label[j]]...)
        j == 1 ? distrib = distrib[distrib .> 1e-3] : distrib = distrib[distrib .> 1e-5]
        flierprops = Dict("marker"=>"o", 
                        "markerfacecolor"=>COLORMODELS[i], 
                        "markersize"=>3,
                        "linestyle"=>"none", "markeredgecolor"=>COLORMODELS[i])

        bplot = ax.boxplot([distrib], positions = [i], 
                            patch_artist=true,  # fill with color
                            showfliers = false,
                            flierprops = flierprops,
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
    # log model plot
    i = 5
    df = filter(r -> (r.Model == model_nul) && r.ΔBIC < 10, df_arr_sorted)
    distrib = vcat(df[:,median_label[j]]...)
    # j == 1 ? distrib = distrib[distrib .> 1e-3] : distrib = distrib[distrib .> 1e-5]
    flierprops = Dict("marker"=>"o", 
                    "markerfacecolor"=>COLORMODELS[i], 
                    "markersize"=>3,
                    "linestyle"=>"none", "markeredgecolor"=>COLORMODELS[i])

    bplot = ax.boxplot([distrib], positions = [i], 
                        patch_artist=true,  # fill with color
                        showfliers = false,
                        flierprops = flierprops,
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
    
    # ax.set_yscale("log")
    if j ∈ [1,3]
        ax.set_xticks(1:length(model_labels))
        ax.set_xticklabels(model_labels)
        if j == 3
            ax.set_ylabel(median_ylabel[j], y = 1 )
        else
            ax.set_ylabel(median_ylabel[j] )
        end
    end
end

_let = ["A","B","C","D"]
for (i,ax) in enumerate(axs[1:2])
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
fig.subplots_adjust(hspace=0.05,wspace = 0.5)  # adjust space between axes
ax_2.set_ylim(0.015, 0.3)  # outliers only
ax_3.set_ylim(-0.003, 0.012)  # most of the data

ax_2.spines.bottom.set_visible(false)
ax_3.spines.top.set_visible(false)
ax_2.xaxis.tick_top()
ax_2.tick_params(labeltop=false, top=false)  # don't put tick labels at the top
ax_3.xaxis.tick_bottom()
display(fig)

# Now, let's turn towards the cut-out slanted lines.
# We create line objects in axes coordinates, in which (0,0), (0,1),
# (1,0), and (1,1) are the four corners of the axes.
# The slanted lines themselves are markers at those locations, such that the
# lines keep their angle and position, independent of the axes size or scale
# Finally, we need to disable clipping.

d = .4  # proportion of vertical to horizontal extent of the slanted line
kwargs = Dict(:marker=>[(-1, -d), (1, d)], :markersize=>12,
              :linestyle=>"none", :color=>"k", :mec=>"k", :mew=>1, :clip_on=>false)
ax_2.plot([0, 1], [0, 0]; transform=ax_2.transAxes, kwargs...)
ax_3.plot([0, 1], [1, 1]; transform=ax_3.transAxes, kwargs...)

# fig.tight_layout()
display(fig)
fig.savefig("figure_aggregate_countries_rs_bs.png", dpi= 300)
