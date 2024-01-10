include("vanishing_grain_FD.jl")
include("vanishing_grain_FE.jl")
include("postprocess.jl")

radius_FD, radius_CG, t, radius_theory = calculate_radius(sol_FD,sol_CG,tspan,Δx)

using Plots; pgfplotsx()
default(titlefont = (18, "times"), legendfont = (12, "times"), guidefont = (12, "times"),
        tickfont = (12, "times"))

n = length.(prob_FD.grid.domain)

initial = plot(prob_FD.grid.domain...,reshape(sol_FD[1][:],n),st = :surface,
    camera = (50,30), c = :viridis, title = "Phase field at t=0s", zlabel = "ϕ",
    xlabel = "x[m]", ylabel = "y[m]")
savefig(initial,"IC.pdf")

final = plot(prob_FD.grid.domain...,reshape(sol_FD[end][:],n),
    st = :surface, camera = (50,30),c = :viridis, title = "Phase field at t=$(tspan[end])",
    zlabel = "ϕ", xlabel = "x[m]", ylabel = "y[m]")
savefig(final,"final.pdf")

radius = plot(t, radius_theory, label = "Analytic solution", xlabel = "Time [s]",
            ylabel = "Radius [m]", legend = :topright)
plot!(radius,sol_FD.t, radius_FD, label = "Finite Difference solution")
scatter!(radius,sol_CG.t, radius_CG, label = "Finite Element solution")
savefig(radius,"radius.pdf")
