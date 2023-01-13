
# this is solution to du/dt = u(r - bu)
@. log_growth(t, p, u0) = (p[1] * u0 * exp(p[1] * t)) / (p[1] + p[2] * u0 * (exp(p[1]*t) - 1f0))

# this is solution to du/dt = r u(1 - bu)
@. log_growth2(t, p, u0) = (u0 * exp(p[1] * t)) / (1 + p[2] * u0 * (exp(p[1]*t) - 1f0))


# not working
# @. log_growth2(t, p, u0) = (u0 * exp(p[1] * t)) / (1 + p[2] * u0 * (exp(p[1]*t) - 1f0))

function init_r_b(data, ts, p0 = [1f-3, 10f0])
    r = similar(data[:,1])
    b = similar(data[:,1])
    for i in 1:size(data, 1)
        fit = curve_fit((t,p) -> log_growth(t, p, data[i,1]), ts, data[i,:], p0, lower = [-0f0, 0f0] .|> eltype(p0))
        r[i] = fit.param[1]
        b[i] = fit.param[2]
    end
    return r, b
end

function init_r_b2(data, ts, p0 = [1f-3, 10f0])
    r = similar(data[:,1])
    b = similar(data[:,1])
    for i in 1:size(data, 1)
        fit = curve_fit((t,p) -> log_growth2(t, p, data[i,1]), ts, data[i,:], p0, lower = [-0f0, 0f0] .|> eltype(p0))
        r[i] = fit.param[1]
        b[i] = fit.param[2]
    end
    return r, b
end

"""
init_r_b_u0(data, ts; r0 = 1f-3, b0 = 10f0)

estimating u0, r and b. 
log_growth2 is used as the underling model
"""
function init_r_b_u0(data, ts; r0 = 1f-3, b0 = 10f0)
    u0 = similar(data[:,1])
    r = similar(data[:,1])
    b = similar(data[:,1])
    RSS = 0.
    for i in 1:size(data, 1)
        p0 = [max(0f0,data[i,1]), r0, b0]
        fit = curve_fit((t,p) -> log_growth2(t, p[2:3], p[1]), ts, data[i,:], p0, lower = zeros(eltype(p0),3))
        u0[i] = fit.param[1]
        r[i] = fit.param[2]
        b[i] = fit.param[3]
        RSS += sum(fit.resid.^2)
    end
    return u0, r, b, RSS
end

# not working
# function init_r_b_u0_log(data, ts; r0 = 1f-1, b0 = 10f0)
#     u0 = similar(data[:,1])
#     r = similar(data[:,1])
#     b = similar(data[:,1])
#     RSS = 0.
#     for i in 1:size(data, 1)
#         p0 = [max(0f0,data[i,1]), r0, b0]
#         tsi = ts[data[i,:] .> 0.]
#         data_i =  log.(data[i, data[i,:] .> 0.])
#         fit = curve_fit((t,p) -> loglog_growth2(t, p[2:3], p[1]), tsi, data_i, p0, lower = 1f-2*ones(eltype(p0),3), upper = lower = 10*ones(eltype(p0),3))
#         u0[i] = fit.param[1]
#         r[i] = fit.param[2]
#         b[i] = fit.param[3]
#         RSS += sum(fit.resid.^2)
#     end
#     return u0, r, b, RSS
# end