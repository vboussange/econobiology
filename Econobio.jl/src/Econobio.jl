#=
Module containing all utils functions
=#
__precompile__(false)
module Econobio

    using Interpolations, LsqFit
    using Statistics
    
    using UnPack, JLD2, FileIO, InlineStrings
    using LightGraphs, SimpleWeightedGraphs

    using DocStringExtensions

    using SciMLBase, Bijectors
    using OrdinaryDiffEq, DiffEqSensitivity

    using MiniBatchInference

    using LinearAlgebra

    using Requires

    using DataFrames

    include("models.jl")
    include("composable_model.jl")

    function __init__()
        @require PyPlot="d330b81b-6aea-500a-939a-2ce795aea3ee" begin 
            include("cb.jl")
            export cb, COLOR_PALETTE_1DIG
        end
    end

    include("extract_data_for_simu.jl")
    include("grad_descent.jl")
    include("lsq_fit.jl")
    include("result_econobio.jl")
    include("constant.jl")
    include("utils.jl")

    export get_data_country, get_economics
    export loss_log 
    export init_r_b, init_r_b2, init_r_b_u0 #init_r_b_u0_log
    export AbstractModel,ModelParams,ModelLog,Modelα,Modelμ,Modelδ,Modelαμ,Modelαδ,Modelμδ,Modelαμδ,
        ComposableModel, Modelμonly, ModelμonlyProducts, Modelδonly, ModelδonlyProducts,ModelαProducts,
        simulate,get_prob,getr,getb, getμ, getδ, getα
    export plot_loss
    export ResultEconobio, construct_result, loglikelihood, estimate_σ,
        get_var_covar_matrix, compute_cis, compute_cis_normal, compute_cis_lognormal,
        name, R2
    export YEARS_IDX, COUNTRIES_ALL, SITC_LABELS_1DIG
    export Squared, Abs, NegAbs

end