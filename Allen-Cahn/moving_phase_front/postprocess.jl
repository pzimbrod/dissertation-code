using Plots; pgfplotsx()

plot(sol_FD[end][:], title = "Phase field at t = $(tspan[end])s", label="t = 100")
plot!(sol_FD[1][:], label="t = 0s")

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

t = range(tspan[1], tspan[end], length = 500) |> collect
position_theory = positions_FD[1] .+ M / Γ * μ₀ .* t;

plot([t,sol_FD.t], [position_theory,positions_FD], label = ["analytic solution" "FD solution"],
        xlabel = "Time [s]", ylabel = "Position of phase front [m]", legend = :topleft)
