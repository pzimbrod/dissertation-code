using LinearAlgebra

include("parameters.jl")

function IC!(u,mesh,R₀,r,parameters)
    x₀, y₀ = R₀
    @views x, y = mesh.domain[1], mesh.domain[2]
    ξ = parameters["ξ"]
    @inbounds for i in axes(u,1), j in axes(u,2)
        u[i,j] = 0.5 * (1. - tanh( (sqrt((x[j] - x₀)^2 + (y[i] - y₀)^2) - r) / ξ))
    end
    return u
end


function apply_rhs!(dϕ,ϕ,p)
    ξ,μ₀,Γ,M = p["ξ"],p["μ₀"],p["Γ"],p["M"]
    @. dϕ += M * (μ₀ / (3. * Γ * ξ) * dh(ϕ) - 2.0 / ξ^2 * dg(ϕ))
end

include("../Common/FDProblem.jl")

using OrdinaryDiffEq

function solve_allenCahn_FD(order,grid,bc,operator,scheme,rhs!,IC!,params,tspan)
    x,y,Δx,Δy = grid["x"],grid["y"],grid["Δx"],grid["Δy"]

    prob_FD = FDProblem(
        order,
        bc,
        operator,
        scheme,
        rhs!,
        params,
        (x,y),
        (Δx,Δy)
    )

    R₀ = (0.,0.)
    r = 20.

    IC!(prob_FD.u,prob_FD.grid,R₀,r,params)

    ode_prob_FD = ODEProblem(prob_FD, prob_FD.u, tspan)
    sol_FD = solve(ode_prob_FD, Rodas4P2(autodiff=false), saveat = 10.0)

    return prob_FD, sol_FD
end

tspan = (0,100)
order = 1
bc = NeumannBC()
operator = Laplacian()
stencil = CentralDifference()
prob_FD, sol_FD = solve_allenCahn_FD(order,grid,bc,operator,stencil,apply_rhs!,IC!,p,tspan)

using BenchmarkTools
bench = @benchmarkable solve_allenCahn_FD(order,grid,bc,operator,stencil,
                                        apply_rhs!,IC!,p,tspan)
# Uncomment this if you would like to run the benchmarks
#run(bench,samples=100,evals=100,seconds=500)
