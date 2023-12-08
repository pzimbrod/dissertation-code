abstract type BoundaryCondition end
struct NeumannBC    <: BoundaryCondition end
struct DirichletBC  <: BoundaryCondition end
struct RobinBC      <: BoundaryCondition end

function apply_neumann_bc!(ϕ::AbstractMatrix)
    @views begin
        # left
        ϕ[1,:] .= ϕ[2,:]
        # Right
        ϕ[end,:] .= ϕ[end-1,:]
        # Top
        ϕ[:,end] .= ϕ[:,end-1]
        # Bottom
        ϕ[:,1] .= ϕ[:,2]
    end
    return nothing
end

function apply_neumann_bc!(ϕ::AbstractVector)
    @views begin
        # left
        ϕ[1] = ϕ[2]
        # Right
        ϕ[end] = ϕ[end-1]
    end
    return nothing
end

select_bc(::NeumannBC) = apply_neumann_bc!
