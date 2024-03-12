#Parameters of the model
p = Dict(
"Γ"  => 50.0,                 # [J/m^2] interface energy
"ξ"  => 4.0,                  # [m] phasefield interface width
"M"  => 1.0,                  # [m^2/s] kinetic coefficient
"μ₀" => 0.0,                  # [J/m^3] bulk energy density difference (driving force)
)

x = (0.,50.)
y = (0.,50.)
Δx = 1.0
Δy = 1.0

grid = Dict(
    "x" => x,
    "y" => y,
    "Δx" => Δx,
    "Δy" => Δy
)

function dg(ϕ)
    return 2.0 * ϕ * (1.0 - ϕ) * (1.0 - 2.0 * ϕ)
end

function dh(ϕ)
    return 6.0 * ϕ * (1.0 - ϕ)
end
