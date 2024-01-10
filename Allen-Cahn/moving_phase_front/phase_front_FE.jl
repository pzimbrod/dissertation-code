using LinearAlgebra

include("parameters.jl")

function IC!(ϕ,mesh,p)
    x₀, ξ = p["x₀"], p["ξ"]
    @views x = mesh.vertices
    @. ϕ = 0.5 * (1.0 - tanh((x - x₀)/ξ))
    return nothing
end

function apply_source_terms!(coeffs,ϕ,p)
    ξ,μ₀,Γ,M = p["ξ"],p["μ₀"],p["Γ"],p["M"]
    @. coeffs = M * (μ₀ / (3. * Γ * ξ) * dh(ϕ) - 2.0 / ξ^2 * dg(ϕ))
end

include("../Common/FEProblem.jl")

function solve_allenCahn_CG(order,primitive,eltype,grid,IC!,rhs!,params)
    x,Δx = grid["x"],grid["Δx"]

    prob_CG = FEProblem(
        primitive,
        eltype,
        x,
        Δx,
        order,
        rhs!,
        params
    )

    IC!(prob_CG.u,prob_CG.Mesh,p)

    ode_prob_CG = generate_ODEProblem(prob_CG,tspan)
    alg_choice = eltype == Spectral() ? ROCK4() : Rodas4P2(autodiff=false)
    sol_CG = solve(ode_prob_CG, alg_choice, saveat = 10.0)

    return sol_CG
end

using DifferentialEquations
tspan = (0,100)
order = 1
primitive = Interval()
# Change to Lagrange() to assemble linear system
# with full accuracy in quadrature
el_type = Spectral()
sol_CG = solve_allenCahn_CG(order,primitive,el_type,grid,IC!,apply_source_terms!,p)

"""
using BenchmarkTools
bench = @benchmarkable solve_allenCahn_CG(order,primitive,el_type,grid,
                                            IC!,apply_source_terms!,p)
run(bench,samples=100,evals=100,seconds=500)
"""
