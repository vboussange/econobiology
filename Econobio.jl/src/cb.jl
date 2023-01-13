#=
Call back function for MiniBatchInference
=#
using PyPlot
_cmap = PyPlot.cm.get_cmap("tab20c", 20) # plotting only nine countrues
const COLOR_PALETTE_1DIG = [_cmap(i-1) for i in 2:2:20];

_cmap = PyPlot.cm.get_cmap("tab20", 20) # only for cb
color_palette = [_cmap(i) for i in 1:20]


function cb(θs, p_trained, losses, pred, ranges, data_set_caggreg, tsteps; verbose = true)
    #########################
    ### plotting ############
    #########################
    if losses[end] < Inf
        N = size(data_set_caggreg,1)

        prod_to_plot = sortperm(sum(data_set_caggreg,dims=2)[:], rev=true)[1:min(N,20)] #plotting only 20 biggest sectors

        PyPlot.close("all")

        fig, axs = plt.subplots(2, figsize = (5, 7)) # loss, params convergence and 2 time series (ensemble problems)


        # plotting loss
        ax = axs[1]
        ax.plot(1:length(losses), losses, c = "tab:blue", label = "Loss")
        ax.legend()
        ax.set_yscale("log")
        ax.set_xlabel("Iterations"); ax.set_ylabel("Loss")
        ax.set_yticklabels([""])
        # plotting time series

        ax = axs[2]
        for g in 1:length(ranges)
            _pred = pred[g]
            _tsteps = tsteps[ranges[g]]
            for (i,p) in enumerate(prod_to_plot)
                ax.plot(_tsteps, _pred[p,:,1], color = color_palette[i])
            end
            for (i,p) in enumerate(prod_to_plot)
                ax.scatter(_tsteps, 
                            data_set_caggreg[p,ranges[g],1], 
                            # label = sitc_fulllabels[p], 
                            s=5., 
                            color = color_palette[i])
            end
        end
        years = 1962 .+ tsteps
        ax.set_xticks(tsteps[1:3:end])
        ax.set_xticklabels(Int.(years)[1:3:end], rotation=45)
        # fig.legend(
        #         loc="upper center", 
        #         bbox_to_anchor=(0.3, 2.),
        #         fancybox=true,
        #         # shadow=true
        #         )

        ax.set_yscale("log")
        fig.tight_layout()

        display(fig)
    end

    return false

end