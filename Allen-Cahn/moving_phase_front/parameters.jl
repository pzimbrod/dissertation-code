#Parameters of the model
const Γ  = 1.0                  # [J/m^2] interface energy
const ξ  = 1.5                  # [m] phasefield interface width
const M  = 1.0                  # [m^2/s] kinetic coefficient
const μ₀ = 0.1                  # [J/m^3] bulk energy density difference (driving force)


const Δx = 1.0
const x₀ = 20.0

grid = Dict(
    "x" => (0.,100.),
    "Δx" => Δx
)

p = Dict(
    "Γ" => Γ,
    "ξ" => ξ,
    "M" => M,
    "μ₀" => μ₀,
    "x₀" => x₀,
)

function dg(ϕ)
    return 2.0 * ϕ * (1.0 - ϕ) * (1.0 - 2.0 * ϕ)
end

function dh(ϕ)
    return 6.0 * ϕ * (1.0 - ϕ)
end
