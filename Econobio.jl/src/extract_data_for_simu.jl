#=
This script reads harvard data
and returns it in an ad hoc format
for simulations, 
normalised by population size over time
=#

"""
$(SIGNATURES)

returns `ts, N, _scale, data_set_caggreg, uworld, Σ`
"""
function get_data_country(country, ndigits; todiscard = ["ZZ", "unspecified", "93", "91"])
    dir = @__DIR__
    data_set, country_labels, sitc_labels, sitc_fulllabels = cd( () -> load("training_data_export_+world_1digit.jld2", "data_set", "country_labels", "sitc_labels", "sitc_fulllabels"), dir*"/../../data/training_data/")
    Pc = cd( () -> load("data_set_pop_+world.jld2", "data_set_pop"), dir*"/../../data/training_data/")

    # first discarding products of no interest
    padding_p_to_discard = .!(sitc_fulllabels .∈ Ref(todiscard))
    data_set = data_set[:,padding_p_to_discard,:]
    sitc_labels = sitc_labels[padding_p_to_discard]
    sitc_fulllabels = sitc_fulllabels[padding_p_to_discard]

    Xcp = permutedims(data_set, [1,3,2]) # countries x time steps x products

    Xc = sum(Xcp[1:end-1,:,:], dims=3) # sum over products, excluding world
    Xp = sum(Xcp[1:end-1,:,:], dims=1) # sum over countries, excluding world

    # Rcp_pop
    P = sum(Pc[1:end-1,:,:], dims=1) # sum over countries, excluding world
    Rcp_pop = Xcp .* P ./ Pc ./ Xp # revealed comparative advantage overtime, scaled by population

    # Excluding countries, products, years
    idx_c = country_labels .== country

    Rcp_c = Rcp_pop[idx_c, :, :] #selecting RCP for each product over time for the country of interest
    Rcp_c_bool = Rcp_c .>= 1
    padding_rcp = sum(Rcp_c_bool,dims=[1,2]) .> 4 # selecting products for which country has had RCA > 1 in at least 4 year (in Hidalgo2021, it is prescribed to average RCA over 3-5 years)
    padding_rcp = padding_rcp[:]
    sitc_labels = sitc_labels[padding_rcp] # saving to simulate back the dynamics
    sitc_fulllabels = sitc_fulllabels[padding_rcp] # saving to simulate back the dynamics

    # Permutations for adaption to DiffEqFlux
    data_set_caggreg = permutedims(data_set, [2,3,1])[:,:,idx_c] # products x time horizon
    data_set_caggreg = data_set_caggreg[padding_rcp, :, 1]

    # using Plots Plots.plot(data_set_caggreg[:,:,1]')
    data_set_world = sum(permutedims(data_set, [2,3,1])[:, :,end],dims=3)
    data_set_world = data_set_world[padding_rcp,:] .- data_set_caggreg # retrieving national export to world exports

    # normalising by population of the country
    data_set_caggreg .= data_set_caggreg ./ Pc[idx_c, :] 
    #normalising by world population minus population of country considerd
    data_set_world .= data_set_world ./ (Pc[end:end, :] .- Pc[idx_c, :])

    ## Normalising for numerical simulation ##
    # since the data domain is strictly positive
    # we just need to divide by the maximum
    _scale = maximum(data_set_caggreg)
    data_set_caggreg .= data_set_caggreg ./ _scale
    data_set_world .= data_set_world ./ _scale

    ts = (1:size(data_set_caggreg,2)) .- 1e0 
    N = size(data_set_caggreg,1) #number of products

    Σ = var(data_set_caggreg, dims = 2) # not used for now

    # linear interpolation world
    interp_data_set_world = [ LinearInterpolation(ts, data_set_world[i,:]) for i in 1:N]
    uworld(t) = [itp(t) for itp in interp_data_set_world]

    return ts, N, _scale, data_set_caggreg, sitc_labels, sitc_fulllabels, uworld, Σ
end

function get_economics(df_results)
    dir = @__DIR__
    eci_ar, pci_ar, gdp_ar, country_labels = cd( () -> load("eci_pci.jld2", "eci_array", "pci_array", "gdp_array", "country_labels"), dir*"/../../data/training_data/")
    data_set_pop =cd( () -> load("data_set_pop.jld2", "data_set_pop"), dir*"/../../data/training_data/")

    gdp_ar ./= data_set_pop
    # pci_ar = pci_ar[padding,:] # pci not needed
    eci_last = [last(eci_ar[i,:]) for i in 1:size(eci_ar,1)]
    eci_first = [first(eci_ar[i,:]) for i in 1:size(eci_ar,1)]
    eci_median = [median(eci_ar[i,:]) for i in 1:size(eci_ar,1)]
    gdp_last = [last(gdp_ar[i,:]) for i in 1:size(eci_ar,1)]
    gdp_first = [first(gdp_ar[i,:]) for i in 1:size(eci_ar,1)]
    gdp_median = [median(gdp_ar[i,:]) for i in 1:size(eci_ar,1)]

    df = DataFrame("country" => country_labels, 
                    "eci_last" => eci_last,
                    "eci_first" => eci_first,
                    "eci_median" => eci_median,
                    "gdp_last" => gdp_last,
                    "gdp_first" => gdp_first,
                    "gdp_median" => gdp_median,)
    df = innerjoin(df, df_results, on=:country )
    return df
end