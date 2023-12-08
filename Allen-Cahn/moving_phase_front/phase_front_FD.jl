using LinearAlgebra
using OrdinaryDiffEq

include("parameters.jl")

function integral(ϕ, Δx)
    return sum(ϕ) * Δx^2
end


function IC!(ϕ,mesh,p)
    x₀, ξ = p["x₀"], p["ξ"]
    @views x = mesh.domain[1]
    @. ϕ = 0.5 * (1.0 - tanh((x - x₀)/ξ))
    return nothing
end

function apply_rhs!(dϕ,ϕ,p)
    ξ,μ₀,Γ,M = p["ξ"],p["μ₀"],p["Γ"],p["M"]
    @. dϕ += M * (μ₀ / (3. * Γ * ξ) * dh(ϕ) - 2.0 / ξ^2 * dg(ϕ))
end

include("../Common/FDProblem.jl")

function solve_allenCahn_FD(order,grid,bc,operator,scheme,rhs!,IC!,params,tspan)
    x,Δx = grid["x"],grid["Δx"]

    prob_FD = FDProblem(
        order,
        bc,
        operator,
        scheme,
        rhs!,
        params,
        (x,),
        (Δx,)
    )

    IC!(prob_FD.u,prob_FD.grid,params)

    ode_prob_FD = ODEProblem(prob_FD, prob_FD.u, tspan)
    sol_FD = solve(ode_prob_FD, ROCK4(), saveat = 10.0)

    return sol_FD
end


tspan = (0,100)

order = 1
bc = NeumannBC()
operator = Laplacian()
stencil = CentralDifference()

sol_FD = solve_allenCahn_FD(order,grid,bc,operator,stencil,
                            apply_rhs!,IC!,p,tspan)

using BenchmarkTools
bench = @benchmarkable solve_allenCahn_FD(order,grid,bc,operator,stencil,
                                        apply_rhs!,IC!,p,tspan)
run(bench,samples=100,evals=100,seconds=500)

include("postprocess.jl")
