#=
Generating .jld2 from harvard dataset to feed model training

In this script we aggregate the 2 digits data to a 1 digit data,
to bring a robustness argument
=#

# using Pkg; Pkg.activate(".")
cd(@__DIR__)
using DataFrames
using CSV
using ProgressMeter
data_path = "../harvard-atlas/country_sitcproductsection_year.csv"

df = DataFrame(CSV.File(data_path))
df_labels = DataFrame(CSV.File("../harvard-atlas/sitc_product.csv"))
sort!(df,[:year,:location_code,:sitc_product_code])

year_labels = sort(unique(df.year))
country_labels = sort(unique(df.location_code))
sitc_labels = sort(unique(df.sitc_product_code))
sitc_fulllabels = [first(df_labels[df_labels[:,2] .== l,3]) for l in sitc_labels]

@assert length(sitc_fulllabels) == length(sitc_labels)

data_set = zeros(length(country_labels),
                length(sitc_labels),
                length(year_labels),
                )

progr = Progress(length(year_labels), showspeed = true, barlen = 10)
idx_rows = collect(1:size(df,1))
global idf = 1
for k in 1:length(year_labels)
    y = year_labels[k]
    for (i,c) in enumerate(country_labels)
        for (j,s) in enumerate(sitc_labels)
            _found = (df.location_code[idf] .== c) .& (df.sitc_product_code[idf] .== s) .& (df.year[idf] .== y)
            if _found
                data_set[i, j, k] = df[idf, :export_value]
                global idf += 1
                # println("added value")
                if idf > size(df,1)
                    break
                end
            end
        end
    end
    next!(progr)
end
using FileIO

save("training_data_export_+world_1digit.jld2", Dict("data_set"=>vcat(data_set,sum(data_set,dims=1)), 
        "year_labels"=> year_labels,
        "country_labels"=> [country_labels; "WLD"],
        "sitc_labels" => sitc_labels,
        "sitc_fulllabels" => sitc_fulllabels))