using LinearAlgebra

include("parameters.jl")

function IC!(ϕ,mesh,R₀,r,parameters)
    x₀, y₀ = R₀
    @views x, y = mesh.vertices[:,1], mesh.vertices[:,2]
    ξ = parameters["ξ"]
    @. ϕ = 0.5 * (1. - tanh( (sqrt((x - x₀)^2 + (y - y₀)^2) - r) / ξ))
    return nothing
end

function apply_source_terms!(coeffs,ϕ,p)
    ξ,μ₀,Γ,M = p["ξ"],p["μ₀"],p["Γ"],p["M"]
    @. coeffs = M * (μ₀ / (3. * Γ * ξ) * dh(ϕ) - 2.0 / ξ^2 * dg(ϕ))
end

include("../Common/FEProblem.jl")

function solve_allenCahn_CG(order,primitive,eltype,grid,IC,rhs!,params)
    x,y,Δx,Δy = grid["x"],grid["y"],grid["Δx"],grid["Δy"]

    prob_CG = FEProblem(
        primitive,
        eltype,
        x,
        y,
        Δx,
        Δy,
        order,
        rhs!,
        params
    )

    R₀ = (0.,0.)
    r = 20.

    IC!(prob_CG.u,prob_CG.Mesh,R₀,r,p)

    ode_prob_CG = generate_ODEProblem(prob_CG,tspan)
    alg_choice = eltype == Spectral() ? ROCK4() : Rodas4P2(autodiff=false)
    sol_CG = solve(ode_prob_CG, alg_choice, saveat = 10.0)

    return sol_CG
end

using DifferentialEquations
tspan = (0,100)
order = 1
primitive = Quadrilateral()
el_type = Lagrange()
sol_CG = solve_allenCahn_CG(order,primitive,el_type,grid,IC!,apply_source_terms!,p)

using BenchmarkTools
bench = @benchmarkable solve_allenCahn_CG(order,primitive,el_type,grid,
                                            IC!,apply_source_terms!,p)
# Uncomment this if you would like to run the benchmarks
#run(bench,samples=100,evals=100,seconds=500)
