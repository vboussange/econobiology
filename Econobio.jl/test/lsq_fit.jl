# testing
using OrdinaryDiffEq, Statistics
dim_prob = 10

@testset "init_r_b" begin
    function model(u, p)
        r = p[1:dim_prob]
        b = p[dim_prob+1:2*dim_prob]
        u .* (r - u .* b)
    end

    ## Data generation
    function dudt(du, u, p, t)
        ũ = max.(u, 0f0)
        du .= model(ũ, p)
    end

    r =  0.2 .*rand(Float32, dim_prob)
    b =  0.2 .*rand(Float32, dim_prob)
    p_init = [r; b]

    # Define the problem
    u0 = rand(dim_prob)
    ts = collect(0f0:60f0)
    tspan = (ts[1], ts[end])
    prob_nn = ODEProblem(dudt, u0, tspan)
    sol = solve(prob_nn, Vern7(), p = p_init, saveat = 1f0)
    data = Array(sol) .|> Float32
    # using Plots
    # plot(sol)

    r_lsq, b_lsq= init_r_b(data, ts)

    @test median((r_lsq - r).^2) < 1f-5
    @test median((b_lsq - b).^2)  < 1f-5
end


@testset "init_r_b2" begin
    function model(u, p)
        r = p[1:dim_prob]
        b = p[dim_prob+1:2*dim_prob]
        r .* u .* (ones(eltype(u),size(u)) - u .* b)
    end

    ## Data generation
    function dudt(du, u, p, t)
        ũ = max.(u, 0f0)
        du .= model(ũ, p)
    end

    r =  0.2 .*rand(Float32, dim_prob)
    b =  0.2 .*rand(Float32, dim_prob)
    p_init = [r; b]

    # Define the problem
    u0 = rand(dim_prob)
    ts = collect(0f0:60f0)
    tspan = (ts[1], ts[end])
    prob_nn = ODEProblem(dudt, u0, tspan)
    sol = solve(prob_nn, Vern7(), p = p_init, saveat = 1f0)
    data = Array(sol) .|> Float32

    r_lsq, b_lsq= init_r_b2(data, ts)

    @test median((r_lsq - r).^2) < 1f-5
    @test median((b_lsq - b).^2)  < 1f-5
end

@testset "init_r_b2_u0" begin
    function model(u, p)
        r = p[1:dim_prob]
        b = p[dim_prob+1:2*dim_prob]
        r .* u .* (one(eltype(u)) .- u .* b)
    end

    ## Data generation
    function dudt(du, u, p, t)
        ũ = max.(u, 0f0)
        du .= model(ũ, p)
    end

    r =  0.2 .*rand(Float32, dim_prob)
    b =  0.2 .*rand(Float32, dim_prob)
    p_init = [r; b]

    # Define the problem
    u0 = rand(dim_prob)
    ts = collect(0f0:60f0)
    tspan = (ts[1], ts[end])
    prob_nn = ODEProblem(dudt, u0, tspan)
    sol = solve(prob_nn, Vern7(), p = p_init, saveat = 1f0)
    data = Array(sol) .|> Float32

    u_lsq, r_lsq, b_lsq= init_r_b_u0(data, ts)

    @test median((data[:,1] .- u0).^2) < 1f-5
    @test median((r_lsq - r).^2) < 1f-5
    @test median((b_lsq - b).^2)  < 1f-5


    # not working 
    # u_lsq, r_lsq, b_lsq= init_r_b_u0_log(data, ts)

    # @test median((data[:,1] .- u0).^2) < 1f-5
    # @test median((r_lsq - r).^2) < 1f-5
    # @test median((b_lsq - b).^2)  < 1f-5
end