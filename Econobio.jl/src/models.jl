#=
Here we define the models as structures, 
which allows to define generic models
which can be declined for systems with a
varying number `N` of entities.

It also allows to store default solving algorithms,
so that they can be used off the shelf with minimum
rewriting.

To define a model, you need the following
## Defining the model
```
struct Modelμonly <: AbstractModel
    mp::ModelParams
end
```

## Defining a constructor, with a `Stacked` of Bijectors
in order to constrain the parameter space for each parameters used
(see https://github.com/TuringLang/Bijectors.jl)
```
function Modelμonly(mp, dists)
    @assert length(dists) == 1
    Modelμonly(remake(mp,st=Stacked(dists,[1:1])))
end
```

## Defining the dynamics
```
function (m::Modelμonly)(du, u, p, t)
    @unpack N, lap = m.mp
    ũ = max.(u, 0f0)
    μ = p[end] |>  m.mp.st.bs[1]
    du .= .- μ .* (lap * ũ) 
    return nothing
end
```

## Defining the number of parameters
Base.length(m::Modelμonly) = 1
=#


"""
$(SIGNATURES)

A model takes `mp::ModelParams` as arguments, and a tuple of
bijectors `dists` to transform parameters in order to ensure, e.g., 
positivity during inference.

## Examples
```julia
logmodel = ModelLog(ModelParams(N=N,
                            lap=lap_mf,
                            p=[rinit; binit],
                            u0 = u0init),
                (Identity{0}(),Identity{0}()))
```

Make sure that when simulting `mymodel`, you use 
`inverse(mymodel.mp.st)(p_init)`

For distributions from bijectors, one can use:
- Identity
- Exp
- Squared
- Abs
"""
abstract type AbstractModel end
name(m::AbstractModel) = string(typeof(m))
Base.show(io::IO, cm::AbstractModel) = println(io, "`Model` ", name(cm))

"""
$(SIGNATURES)

Returns `ODEProblem` associated with `m`.
"""
function get_prob(m::AbstractModel;u0 = m.mp.u0, p=m.mp.p)
    # @assert length(u0) == cm.mp.N # this is not necessary true if u0 is a vecor of u0s
    @assert isnothing(m.mp.tspan) != true
    @assert length(p) == length(m)
    # TODO: inverse to be checked
    prob = ODEProblem(m, u0, m.mp.tspan, inverse(m.mp.st)(p))
    return prob
end

"""
$(SIGNATURES)

Simulate model `m` and returns an `ODESolution`. 
"""
function simulate(m::AbstractModel;u0 = m.mp.u0, p=m.mp.p)
    prob = get_prob(m; u0, p)
    sol = solve(prob, m.mp.alg; sensealg = DiffEqSensitivity.ForwardDiffSensitivity(), m.mp.kwargs_sol...)
    return sol
end

# default behavior model
getr(p,m::AbstractModel) = p[1:m.mp.N] .|> m.mp.st.bs[1]
getb(p,m::AbstractModel) = p[m.mp.N+1:2*m.mp.N] .|> m.mp.st.bs[2]
getα(p,::AbstractModel) = NaN
getμ(p,::AbstractModel) = NaN
getδ(p,::AbstractModel) = NaN

# model parameters
Base.@kwdef struct ModelParams{T,P,U0,G,NN,L,U,ST,A,K<: Dict}
    tspan::T = ()
    p::P = [] # parameter vector
    plabel::Vector{String} = String[] # parameter label vector
    u0::U0 = [] # ICs
    g::G = nothing # graph interactions
    N::NN # nb of products
    lap::L = nothing # laplacian
    uworld::U = nothing # input signal world
    st::ST = Identity{0}()
    alg::A = ()
    kwargs_sol::K = Dict()
end

struct ModelLog <: AbstractModel
    mp::ModelParams
end
# dists must be a tuple of distributions from Bijectors.jl
function ModelLog(mp, dists)
    @assert length(dists) == 2
    ModelLog(remake(mp,st=Stacked(dists,[1:mp.N,mp.N+1:2*mp.N])))
end

function (m::ModelLog)(du, u, p, t)
    @unpack N = m.mp
    ũ = max.(u, 0f0)
    r = getr(p, m)
    b = getb(p, m)
    du .= r .* ũ .* (1f0 .- ũ .* b)
    return nothing
end

struct Modelα <: AbstractModel
    mp::ModelParams
end
getα(p,m::Modelα) = p[end] |>  m.mp.st.bs[3]
function Modelα(mp, dists)
    @assert length(dists) == 3
    Modelα(remake(mp,st=Stacked(dists,[1:mp.N,mp.N+1:2*mp.N,2*mp.N+1])))
end

function (m::Modelα)(du, u, p, t)
    T = eltype(u)
    @unpack N = m.mp

    ũ = max.(u, 0f0)
    r = getr(p, m)
    b = getb(p, m)
    α = getα(p, m)
    du .= r .* ũ .* (1f0 .- ũ .* b .+ α * (sum(ũ) .- ũ) / convert(T,N))
    return nothing
end

struct Modelμ <: AbstractModel
    mp::ModelParams
end
getμ(p,m::Modelμ) = p[end] |>  m.mp.st.bs[3]
function Modelμ(mp, dists)
    @assert length(dists) == 3
    Modelμ(remake(mp,st=Stacked(dists,[1:mp.N,mp.N+1:2*mp.N,2*mp.N+1])))
end

function (m::Modelμ)(du, u, p, t)
    @unpack N, lap = m.mp
    ũ = max.(u, 0f0)
    r = getr(p,m)
    b = getb(p,m)
    μ = getμ(p,m)
    du .= r .* ũ .* (1f0 .- ũ .* b) .- μ .* (lap * ũ) 
    return nothing
end

struct Modelδ <: AbstractModel
    mp::ModelParams
end
getδ(p,m::Modelδ) = p[end] |> m.mp.st.bs[3]
function Modelδ(mp, dists)
    @assert length(dists) == 3
    Modelδ(remake(mp,st=Stacked(dists,[1:mp.N,mp.N+1:2*mp.N,2*mp.N+1])))
end

function (m::Modelδ)(du, u, p, t)
    @unpack N, uworld = m.mp

    ũ = max.(u, 0f0)
    r = getr(p,m)
    b = getb(p,m)
    δ = getδ(p,m)
    du .= r .* ũ .* (1f0 .- ũ .* b) .+ δ .* (uworld(t) - ũ) 
    return nothing
end

struct Modelαμ <: AbstractModel
    mp::ModelParams
end
getα(p,::Modelαμ) = p[end-1]
getμ(p,m::Modelαμ) = p[end] |> m.mp.st.bs[4]
function Modelαμ(mp, dists)
    @assert length(dists) == 4
    Modelαμ(remake(mp,st=Stacked(dists,[1:mp.N,mp.N+1:2*mp.N,2*mp.N+1,2*mp.N+2])))
end

function (m::Modelαμ)(du, u, p, t)
    T = eltype(u)
    @unpack N, lap = m.mp
    ũ = max.(u, 0f0)
    r = getr(p,m)
    b = getb(p,m)
    α = getα(p,m)
    μ = getμ(p,m)
    du .= r .* ũ .* (1f0 .- ũ .* b .+ α * (sum(ũ) .- ũ) / convert(T,N)) .- μ .* (lap * ũ)
    return nothing
end

struct Modelμδ <: AbstractModel
    mp::ModelParams
end
getμ(p,m::Modelμδ) = p[end-1]|> m.mp.st.bs[3]
getδ(p,m::Modelμδ) = p[end] |> m.mp.st.bs[4]
function Modelμδ(mp, dists)
    @assert length(dists) == 4
    Modelμδ(remake(mp,st=Stacked(dists,[1:mp.N,mp.N+1:2*mp.N,2*mp.N+1,2*mp.N+2])))
end

function (m::Modelμδ)(du, u, p, t)
    @unpack N, uworld, lap = m.mp

    ũ = max.(u, 0f0)
    r = getr(p,m)
    b = getb(p,m)
    μ = getμ(p,m)
    δ = getδ(p,m)

    du .= r .* ũ .* (1f0 .- ũ .* b) .- μ * (lap * ũ) .+ δ .* (uworld(t) - ũ)
    return nothing
end

struct Modelαδ <: AbstractModel
    mp::ModelParams
end
getα(p,::Modelαδ) = p[end-1]
getδ(p,m::Modelαδ) = p[end] |> m.mp.st.bs[4]
function Modelαδ(mp, dists)
    @assert length(dists) == 4
    Modelαδ(remake(mp,st=Stacked(dists,[1:mp.N,mp.N+1:2*mp.N,2*mp.N+1,2*mp.N+2])))
end

function (m::Modelαδ)(du, u, p, t)
    T = eltype(u)
    @unpack N, uworld = m.mp

    ũ = max.(u, 0f0)
    r = getr(p,m)
    b = getb(p,m)
    α = getα(p,m)
    δ = getδ(p,m)

    du .= r .* ũ .* (1f0 .- ũ .* b .+ α * (sum(ũ) .- ũ) / convert(T,N)) .+ δ * (uworld(t) - ũ) 
    return nothing
end

struct Modelαμδ <: AbstractModel
    mp::ModelParams
end
getα(p,::Modelαμδ) = p[end-2]
getμ(p,m::Modelαμδ) = p[end-1]|> m.mp.st.bs[4]
getδ(p,m::Modelαμδ) = p[end] |> m.mp.st.bs[5]
function Modelαμδ(mp, dists)
    @assert length(dists) == 5
    Modelαμδ(remake(mp,st=Stacked(dists,[1:mp.N,mp.N+1:2*mp.N,2*mp.N+1,2*mp.N+2,2*mp.N+3])))
end

function (m::Modelαμδ)(du, u, p, t)
    T = eltype(u)
    @unpack N, uworld, lap = m.mp
    ũ = max.(u, 0f0)
    r = getr(p,m)
    b = getb(p,m)
    α = getα(p,m)
    μ = getμ(p,m)
    δ = getδ(p,m)
    du .= r .* ũ .* (1f0 .- ũ .* b .+ α * (sum(ũ) .- ũ) / convert(T,N)) .- μ .* (lap * ũ) .+ δ .* (uworld(t) - ũ) 
    return nothing
end

Base.length(m::ModelLog) = 2 * m.mp.N
for Mod in [Modelα,Modelμ,Modelδ,Modelαμ,Modelαδ,Modelμδ,Modelαμδ]
    @eval Base.length(m::$Mod) = length(string($Mod)) - 5 + 2 * m.mp.N
end


"""
    choose_scenario_random_p(scenario, u0_log_LM, r_log_LM, b_log_LM, dict_ϵ)

return `dudt, u0_init, p_init` with perturbations given by `dict_ϵ`
"""
function choose_scenario(;scenario, r_log_LM, b_log_LM, u0_log_LM, uworld, dict_ϵ, bijector)
    p_init = [r_log_LM; b_log_LM] .|> Float32 |> inverse(bijector)
    u0_init = u0_log_LM .|> Float32 .|> inverse(bijector)
    # adding a small perturbation to parameters
    p_init = p_init + p_init .* randn(Float32, length(p_init)) .* ϵ_log
    u0_init = u0_init + u0_init .* randn(Float32, length(u0_init)) .* ϵ_log

    N = length(u0_init)
    g = complete_graph(N)
    lap_mf = Matrix{Float32}(laplacian_matrix(g)) ./ Float32(N)

    if scenario ∈ ("α", "μ", "δ")
        append!(p_init, randn(Float32,1) .* [dict_ϵ[l] for l in scenario] )
    elseif scenario ∈ ("αμ", "αδ", "μδ")
        append!(p_init, randn(Float32,2) .* [dict_ϵ[l] for l in scenario] )
    elseif scenario == "αμδ"
        dudt = eval(Symbol("dudt_",scenario))
    end
    mp = ModelParams(N=N, lap=lap_mf, uworld = uworld_inv, p = pinit)
    dudt = eval(Symbol("Model",scenario))(mp)
    return dudt
end