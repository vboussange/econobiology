#=
Generate country code table (Table S1).
=#

# using Pkg; Pkg.activate(".")
cd(@__DIR__)
using DataFrames
using CSV
using FileIO
using Latexify

data_path = "../../../data/worldbank/population/API_SP.POP.TOTL_DS2_en_csv_v2_2763937.csv"

df = DataFrame(CSV.File(data_path, header = 5, select = 1:2)) # from 1962 to 2020
df = df[df[:,"Country Code"] .∈ Ref(COUNTRIES_ALL[2:end]),:]
# countries where we do not have data population data:
# COUNTRIES_ALL[2:end][.!(COUNTRIES_ALL[2:end] .∈ Ref(df[:,"Country Code"]))]

open("country_code.csv", "w") do io
    # tab_stats = latexify(df,env=:tabular,fmt="%.7f",latex=false) #|> String
    println(io,"Country Code", ";", "Country Name")

    for r in eachrow(df)
        println(io,r["Country Code"], ";", r["Country Name"]);
    end
end