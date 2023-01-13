using Econobio
using OrdinaryDiffEq, Test, LightGraphs, SimpleWeightedGraphs
using Bijectors: Exp, inverse, Identity
using Random; Random.seed!(2)

@testset "ComposableModel Modelμonly" begin
    N = 10
    tspan = (0., 1.)
    tsteps = range(tspan[1], tspan[end], length=10)
    g = complete_graph(N)
    lap_mf = Matrix{Float32}(laplacian_matrix(g)) ./ Float32(N)

    u_init = rand(N)
    p_init = [rand(2*N); 0.1]

    m1 = ModelLog(ModelParams(N=N,
                    lap=lap_mf, 
                    uworld = t -> zeros(N), 
                    u0 = u_init,
                    tspan = tspan,
                    alg = BS3(),
                    kwargs_sol = Dict(:saveat => tsteps)),
                    (Identity{0}(),Identity{0}()))
    m2 = Modelμonly(ModelParams(N=N,
                    lap=lap_mf, 
                    uworld = t -> zeros(N), 
                    u0 = u_init,
                    tspan = tspan,
                    alg = BS3() ),
                    (Identity{0}(),))
    cm = ComposableModel((m1,m2))


    sol_cm = simulate(cm, p = p_init)

    m3 = Modelμ(ModelParams(N=N,
                lap=lap_mf, 
                uworld = t -> zeros(N), 
                u0 = u_init,
                tspan = tspan,
                alg = BS3(),
                kwargs_sol = Dict(:saveat => tsteps)),
                (Identity{0}(),Identity{0}(),Identity{0}()))

    sol_m3 = simulate(m3, p = p_init)

    @test all(Array(sol_cm) .≈ Array(sol_m3))
end

@testset "ComposableModel Modelδonly" begin
    N = 10
    tspan = (0., 1.)
    tsteps = range(tspan[1], tspan[end], length=10)
    g = complete_graph(N)
    lap_mf = Matrix{Float32}(laplacian_matrix(g)) ./ Float32(N)

    u_init = rand(N)
    p_init = [rand(2*N); 0.1]

    m1 = ModelLog(ModelParams(N=N,
                    lap=lap_mf, 
                    uworld = t -> zeros(N), 
                    u0 = u_init,
                    tspan = tspan,
                    alg = BS3() ),
                    (Identity{0}(),Identity{0}()))
    m2 = Modelδonly(ModelParams(N=N,
                    lap=lap_mf, 
                    uworld = t -> zeros(N), 
                    u0 = u_init,
                    tspan = tspan,
                    alg = BS3() ),
                    (Identity{0}(),))
    cm = ComposableModel((m1,m2))


    sol_cm = simulate(cm, p = p_init)

    m3 = Modelδ(ModelParams(N=N,
                lap=lap_mf, 
                uworld = t -> zeros(N), 
                u0 = u_init,
                tspan = tspan,
                alg = BS3() ),
                (Identity{0}(),Identity{0}(),Identity{0}()))

    sol_m3 = simulate(m3, p = p_init)

    @test all(Array(sol_cm) .≈ Array(sol_m3))
end

@testset "ModelδonlyProducts" begin
    N = 10
    tspan = (0., 1.)
    tsteps = range(tspan[1], tspan[end], length=10)
    g = complete_graph(N)
    lap_mf = Matrix{Float32}(laplacian_matrix(g)) ./ Float32(N)

    u_init = rand(N)
    p_init = fill(0.1,N)

    m1 = ModelδonlyProducts(ModelParams(N=N,
                    lap=lap_mf, 
                    uworld = t -> zeros(N), 
                    u0 = u_init,
                    tspan = tspan,
                    alg = BS3() ),
                    (Identity{0}(),))
    m2 = Modelδonly(ModelParams(N=N,
                    lap=lap_mf, 
                    uworld = t -> zeros(N), 
                    u0 = u_init,
                    tspan = tspan,
                    alg = BS3() ),
                    (Identity{0}(),))
    
    sol_m1 = simulate(m1, p = p_init)
    sol_m2 = simulate(m2, p = [0.1])

    @test all(Array(sol_m1) .≈ Array(sol_m2))
end

#TODO: Missing test: ModelμonlyProduct

@testset "ModelαProducts" begin
    N = 10
    tspan = (0., 1.)
    tsteps = range(tspan[1], tspan[end], length=10)
    g = complete_graph(N)
    lap_mf = Matrix{Float32}(laplacian_matrix(g)) ./ Float32(N)

    u_init = rand(N)
    p_init1 = [rand(2*N); fill(0.1,N)]
    p_init2 = p_init1[1:2*N+1]

    m1 = ModelαProducts(ModelParams(N=N,
                    lap=lap_mf, 
                    uworld = t -> zeros(N), 
                    u0 = u_init,
                    tspan = tspan,
                    alg = BS3() ),
                    (Identity{0}(),Identity{0}(),Identity{0}()))
    m2 = Modelα(ModelParams(N=N,
                    lap=lap_mf, 
                    uworld = t -> zeros(N), 
                    u0 = u_init,
                    tspan = tspan,
                    alg = BS3() ),
                    (Identity{0}(),Identity{0}(),Identity{0}()))
    
    sol_m1 = simulate(m1, p = p_init1)
    sol_m2 = simulate(m2, p = p_init2)

    @test all(Array(sol_m1) .≈ Array(sol_m2))
end