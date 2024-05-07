function integral(ϕ, Δx)
    return sum(ϕ) * Δx^2
end

function calculate_radius(sol_FD,sol_CG,tspan, Δx)
    radius_FD = zeros(length(sol_FD))
    radius_CG = similar(radius_FD)
    t = range(tspan[1], tspan[end], length = 500) |> collect
    for (i,j) ∈ zip(eachindex(sol_FD),eachindex(sol_CG))
        radius_FD[i] = sqrt(4.0 * integral(sol_FD[i],Δx) / π)
        radius_CG[j] = sqrt(4.0 * integral(sol_CG[j],Δx) / π)
    end
    radius_theory = sqrt.(max.(0.0,(first(radius_FD)^2 .- 2.0 * p["M"] .* t)));
    return radius_FD, radius_CG, t, radius_theory
end
