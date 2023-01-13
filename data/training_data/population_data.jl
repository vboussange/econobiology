# using Pkg; Pkg.activate(".")
cd(@__DIR__)
using DataFrames
using CSV
using FileIO

data_path = "../worldbank/population/API_SP.POP.TOTL_DS2_en_csv_v2_2763937.csv"
country_labels = load("training_data_export_+world_1digit.jld2", "country_labels")[1:end-1] # removing label "WLD"

df = DataFrame(CSV.File(data_path, header = 5, select = [2;7:65])) # from 1962 to 2020
nyears = size(df,2)-1
data_set_pop = zeros(length(country_labels),
                nyears,
                )

countries_nodata = []
for (i,c) in enumerate(country_labels)
    _idf = (df[:,1] .== c)
    if count(_idf) == 1 && count(ismissing.(Array(df[_idf,2:end])[:])) == 0
        data_set_pop[i, :] .= Array(df[_idf,2:end])[:]
    else
        println("error for ", c, ", count = ", count(_idf), ", missing data = ", count(ismissing.(Array(df[_idf,2:end])[:])))
        push!(countries_nodata, c)
    end
end

save("data_set_pop.jld2", Dict("data_set_pop" => data_set_pop, 
                                "countries_nodata" => countries_nodata))

# world = df[df[:,1].== "WLD", 2:end] 
world = sum(data_set_pop,dims=1)
save("data_set_pop_+world.jld2", Dict("data_set_pop" => vcat(data_set_pop,world)))

## analysis
if false
    data_set, country_labels, sitc_labels = load("training_data_export.jld2", "data_set", "country_labels", "sitc_labels")
    data_set_pop = load("data_set_pop.jld2", "data_set_pop")

    country = "CHE"
    countries = [country] # https://atlas.cid.harvard.edu/rankings
    idx_c = country_labels .∈ Ref(countries)
    padding = .!(sitc_labels .∈ Ref(["ZZ", "unspecified", "93", "91"]))
    sitc_labels = sitc_labels[padding]

    # Normalize the data; since the data domain is strictly positive
    # we just need to divide by the maximum
    exact_years = 1:59
    prods = 1:20
    data_set_caggreg = permutedims(data_set .|> Float32, [2,3,1])[:,:,idx_c]
    data_set_caggreg = data_set_caggreg[padding, exact_years, 1] #excluding pandemic, and big drop in ict (2017)
    data_set_caggreg = data_set_caggreg ./ data_set_pop[idx_c, exact_years]

    using PyPlot
    close("all")
    fig = figure()
    for i in prods
        scatter(names(df)[exact_years .+ 1], data_set_caggreg[i,:], label = sitc_labels[i], s = 5.)
    end
    plt.xlabel("years"); plt.ylabel("Export output")
    title(country)
    tick_params(axis = "x", rotation=45)
    plt.legend(sitc_labels[prods])
    gcf()
end