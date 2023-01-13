Base.@kwdef struct ComposableModel{MP,MS,PS} <: AbstractModel
    mp::MP # model params
    models::MS # tuple of abstract models
    param_indices::PS # used for indexing parameters for each submodels
end

""" 
$(SIGNATURES)

A composable model, where the dynamics `du` is the sum of `du_1, ..., du_n`
obtained from `models = (model1,...,modeln)`.

Each model needs to be given initial conditions `u0`, as those are
 used to infer `dutemp`.
"""
function ComposableModel(models::MS)  where {MS <: NTuple{N,AbstractModel} where N}
    @assert all(models[1].mp.N == m.mp.N for m in models)

    shifts = [0;cumsum(length.(models))...]
    param_indices = [shifts[i] + 1:shifts[i+1] for i in 1:length(models)]
    pcompos = vcat([m.mp.p for m in models]...)
    mp = remake(models[1].mp, p = pcompos)

    ComposableModel(mp, models, param_indices)
end
name(cm::ComposableModel) = string(typeof.(cm.models))

import Base.show
Base.show(io::IO, cm::ComposableModel) = println(io, "`ComposableModel` with ", name(cm))

function Base.length(cm::ComposableModel)
    sum(length.(cm.models))
end


function (cm::ComposableModel)(du, u, p, t)
    du .= zero(eltype(du))
    dutemp = similar(du)
    for i in 1:length(cm.models)
        p_i = @view p[cm.param_indices[i]]
        cm.models[i](dutemp, u, p_i, t)
        du .+= dutemp
    end
    return nothing
end

function get_prob(cm::ComposableModel;u0 = cm.mp.u0, p=cm.mp.p)
    # @assert length(u0) == cm.mp.N # this is not necessary true if u0 is a vecor of u0s
    @assert isnothing(cm.mp.tspan) != true
    @assert length(p) == length(cm)
    pinversed = vcat([inverse(cm.models[i].mp.st)(p[cm.param_indices[i]]) for i in 1:length(cm.models)]...)
    # TODO: inverse to be checked
    prob = ODEProblem(cm, u0, cm.mp.tspan, pinversed)
    return prob
end

"""
$(SIGNATURES)

Only diffusion without logistic term
"""
struct Modelμonly <: AbstractModel
    mp::ModelParams
end

function Modelμonly(mp, dists)
    @assert length(dists) == 1
    Modelμonly(remake(mp,st=Stacked(dists,[1:1])))
end

function (m::Modelμonly)(du, u, p, t)
    @unpack N, lap = m.mp
    ũ = max.(u, 0f0)
    μ = p[end] |>  m.mp.st.bs[1]
    du .= .- μ .* (lap * ũ) 
    return nothing
end
Base.length(m::Modelμonly) = 1

"""
$(SIGNATURES)

Only diffusion without logistic term
μ_i defined for all products.
"""
struct ModelμonlyProducts <: AbstractModel
    mp::ModelParams
end
function ModelμonlyProducts(mp, dists)
    @assert length(dists) == 1
    ModelμonlyProducts(remake(mp,st=Stacked(dists,[1:mp.N])))
end

function (m::ModelμonlyProducts)(du, u, p, t)
    @unpack N, lap = m.mp
    ũ = max.(u, 0f0)
    μ = p |>  m.mp.st.bs[1]
    du .= .- μ .* (lap * ũ) 
    return nothing
end
Base.length(m::ModelμonlyProducts) = m.mp.N

"""
$(SIGNATURES)

Only diffusion without logistic term
"""
struct Modelδonly <: AbstractModel
    mp::ModelParams
end

function Modelδonly(mp, dists)
    @assert length(dists) == 1
    Modelδonly(remake(mp,st=Stacked(dists,[1:1])))
end

function (m::Modelδonly)(du, u, p, t)
    @unpack N, uworld = m.mp
    ũ = max.(u, 0f0)
    δ = p[end] |>  m.mp.st.bs[1]
    du .= δ .* (uworld(t) - ũ) 
    return nothing
end
Base.length(m::Modelδonly) = 1

"""
$(SIGNATURES)

Only diffusion without logistic term
δ_i defined for all products.
"""
struct ModelδonlyProducts <: AbstractModel
    mp::ModelParams
end

function ModelδonlyProducts(mp, dists)
    @assert length(dists) == 1
    ModelδonlyProducts(remake(mp,st=Stacked(dists,[1:mp.N])))
end

function (m::ModelδonlyProducts)(du, u, p, t)
    @unpack N, uworld = m.mp
    ũ = max.(u, 0f0)
    δ = p |>  m.mp.st.bs[1]
    du .= δ .* (uworld(t) - ũ) 
    return nothing
end
Base.length(m::ModelδonlyProducts) = m.mp.N

"""
$(SIGNATURES)

Only interactions without logistic term
α_i defined for all products.
"""
struct ModelαProducts <: AbstractModel
    mp::ModelParams
end
getα(p,m::ModelαProducts) = p[2*m.mp.N+1:end] |>  m.mp.st.bs[3]
function ModelαProducts(mp, dists)
    @assert length(dists) == 3
    ModelαProducts(remake(mp,st=Stacked(dists,[1:mp.N,mp.N+1:2*mp.N,2*mp.N+1:3*mp.N])))
end

function (m::ModelαProducts)(du, u, p, t)
    T = eltype(u)
    @unpack N = m.mp

    ũ = max.(u, 0f0)
    r = getr(p, m)
    b = getb(p, m)
    α = getα(p, m)
    du .= r .* ũ .* (1f0 .- ũ .* b .+ α .* (sum(ũ) .- ũ) / convert(T,N))
    return nothing
end
Base.length(m::ModelαProducts) = 3 * m.mp.N