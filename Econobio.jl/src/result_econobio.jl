"""
$(SIGNATURES)

"""
Base.@kwdef struct ResultEconobio{Model<:AbstractModel, RES}
    m::Model
    res::RES
end
import Base.show
Base.show(io::IO, res::ResultEconobio) = println(io, "`ResultEconobio` with model", name(res.m))

function construct_result(m::Model, res::RES) where {Model<:AbstractModel, RES}
    params_trained = remake(m.mp, p=res.p_trained |> m.mp.st)
    return ResultEconobio(typeof(m)(params_trained),res) #/!\ new{Model,RES}( is required! 
end

function construct_result(cm::CM, res::RES) where {CM <: ComposableModel, RES}
    p_trained = res.p_trained
    p = vcat([cm.models[i].mp.st(p_trained[cm.param_indices[i]]) for i in 1:length(cm.models)]...)
    params_trained = remake(cm.mp, p = p)
    cm = remake(cm, mp = params_trained)
    return ResultEconobio(cm, res) #/!\ new{Model,RES}( is required! 
end

function loglikelihood(res::ResultEconobio, 
                        ode_data::Array, 
                        Σ; 
                        loglike_fn = MiniBatchInference.loglikelihood_lognormal , 
                        u0s = res.res.u0s_trained,
                        p = res.res.p_trained) # we take res.res.p_trained because we would have to transform the parameters otherwise

    θ = [u0s...;p] 
    prob = ODEProblem(res.m, u0s[1], res.m.mp.tspan, p)
    loss_fn(data, params, pred, rg) = loglike_fn(data, pred, Σ)
    l, _ = minibatch_loss(θ, ode_data, res.m.mp.kwargs_sol[:saveat], prob, loss_fn, res.m.mp.alg, res.res.ranges; continuity_term=0.)
    return l
end


"""
$(SIGNATURES)

Estimate noise variance, assuming similar noise across all the dimensions of the data.
"""
function estimate_σ(pred::Array, odedata::Array, ::Normal)
    @assert size(pred) == size(odedata)
    RSS = (pred .- odedata).^2
    return sqrt(mean(RSS))
end

function estimate_σ(pred::Array, odedata::Array, ::LogNormal)
    @assert size(pred) == size(odedata)
    logsquare = (log.(pred) .- log.(odedata)).^2
    return sqrt(mean(logsquare[logsquare .< Inf]))
end

function estimate_σ(reseco::ResultEconobio, odedata::Array; noisedistrib=LogNormal(), include_ic = true)
    @assert !isempty(reseco.res.pred) "reseco should have been obtained with `save_pred = true`"
    if include_ic
        odedata = hcat([odedata[:,rg] for rg in reseco.res.ranges]...)
        pred = hcat([reseco.res.pred[i] for i in 1:length(reseco.res.ranges)]...)
    else
        odedata = hcat([odedata[:,rg[2:end]] for rg in reseco.res.ranges]...)
        pred = hcat([reseco.res.pred[i][:,2:end] for i in 1:length(reseco.res.ranges)]...)
    end
    return estimate_σ(pred,odedata,noisedistrib)
end


# TODO: test it !
"""
$(SIGNATURES)

"""
function get_var_covar_matrix(reseco::ResultEconobio, odedata::Array, σ::Number, loglike_fn = MiniBatchInference.loglikelihood_lognormal)
    likelihood_fn_optim(p) = Econobio.loglikelihood(reseco, odedata, σ; p = p, loglike_fn)
    p_trained = reseco.res.p_trained
    numerical_hessian = ForwardDiff.hessian(likelihood_fn_optim, p_trained)
    var_cov_matrix = - inv(numerical_hessian)
    return var_cov_matrix
end

"""
$(SIGNATURES)

Compute confidence intervals, given `var_cov_matrix`, parameters `p` and a confidence level `α`.
"""
function compute_cis(var_cov_matrix::Matrix, p::Vector, α::AbstractFloat)
    ses = sqrt.(diag(var_cov_matrix)) 
    τ = cquantile(Normal(0, 1), α)
    lower = p - τ * ses
    upper = p + τ * ses
    lower, upper
end

"""
$(SIGNATURES)

"""
compute_cis_normal(reseco::ResultEconobio, odedata, p, α, σ) = compute_cis(get_var_covar_matrix(reseco, 
                                                                    odedata,
                                                                    σ, 
                                                                    loglike_fn = MiniBatchInference.loglikelihood_normal), 
                                                                    p, α)

"""
$(SIGNATURES)

"""
compute_cis_lognormal(reseco::ResultEconobio,odedata, p, α, σ) = compute_cis(get_var_covar_matrix(reseco, 
                                                                            odedata,
                                                                            σ, 
                                                                            loglike_fn = MiniBatchInference.loglikelihood_lognormal), 
                                                                            p, α)


"""
$(SIGNATURES)

"""
function R2(odedata, pred, ::LogNormal)
    rsstot = log.(odedata) .- mean(log.(odedata), dims=1)
    rssreg = log.(pred) .- log.(odedata)

    padding = (rsstot .< Inf) .* (rssreg .< Inf)

    vartot = sum(abs2,rsstot[padding])
    var_reg = sum(abs2, rssreg[padding])
    R2 = 1 - var_reg / vartot
    return R2
end

