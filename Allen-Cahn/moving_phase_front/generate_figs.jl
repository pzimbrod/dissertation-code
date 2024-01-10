using Plots; pgfplotsx()
using LaTeXStrings
include("phase_front_FD.jl")
include("phase_front_FE.jl")


default(titlefont = (18, "times"), legendfont = (12, "times"), guidefont = (12, "times"),
        tickfont = (12, "times"))

p1 = plot([sol_FD[1][:] sol_FD[end][:] sol_CG[end][:]],
            label = ["Initital condition" "Finite Difference solution" "Finite Element solution"],
             linewidth = 3,
             linestyle = [:solid :solid :dash],
             xlabel = "Position [m]",
             ylabel = L"Phase field $\phi$",
             legend = :topright)
savefig(p1,"phasefield.pdf")

function integral(ϕ, Δx)
    return sum(ϕ) * Δx^2
end

function pos(ϕ,Δx)
    return integral(ϕ,Δx) / Δx + 0.5 * Δx
end

positions_FD = zeros(length(sol_FD))
@inbounds for i ∈ eachindex(sol_FD)
    positions_FD[i] = pos(sol_FD[i],Δx)
end

positions_CG = zeros(length(sol_CG))
@inbounds for i ∈ eachindex(sol_CG)
    positions_CG[i] = pos(sol_CG[i],Δx)
end

t = range(tspan[1], tspan[end], length = 500) |> collect
position_theory = positions_FD[1] .+ M / Γ * μ₀ .* t;

p = plot([t,sol_FD.t], [position_theory,positions_FD],
        label = ["Analytic solution" "Finite Difference solution"],
        linewidth = 3,
        xlabel = "Time [s]", ylabel = "Position of phase front [m]",
        legend = :topleft)
scatter!(p, sol_CG.t, positions_CG,
        label = "Finite Element solution")
savefig(p,"positions.pdf")
