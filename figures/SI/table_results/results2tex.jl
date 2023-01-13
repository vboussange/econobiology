#=
Generate Table S2.
=#
cd(@__DIR__)
using FileIO, JLD2
using DataFrames, LaTeXStrings
using Latexify

scenario_name = "2022-08-28_fit_countries_all_without_services_1digit_ADAM_1e-3_df_arr_sorted"
@load "../../cleaning_results/$scenario_name.jld2" df_arr_sorted

group_size_init = 21
filter!(row -> row.group_size_init == group_size_init, df_arr_sorted)

model_labels = [L"$\mathcal{M}_{\alpha^+}$", L"$\mathcal{M}_{\alpha^-}$", L"$\mathcal{M}_{\delta}$", L"$\mathcal{M}_{\mu}$", L"$\mathcal{M}_{null}$"]
rename!(df_arr_sorted, Dict("country" => "Country", 
                    "likelihood" => "Log-likelihood",
                    "loss" => "Loss",
                    "std_loss" => "Rel. std. loss",
                    "ΔBIC" => L"\Delta \text{BIC}"))

sort!(df_arr_sorted, :Country)
open("results.tex", "w") do io
    tab_stats = latexify(df_arr_sorted[:,["Country", 
                                        "Model", 
                                        "Loss", 
                                        "Rel. std. loss", 
                                        "R2", 
                                        "Log-likelihood", 
                                        "BIC"]],env=:tabular,fmt="%.7f",latex=false) #|> String
    write(io,tab_stats)
end