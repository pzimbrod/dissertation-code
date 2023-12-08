include("triangulation.jl")
include("FiniteDifference/boundary_condition.jl")
include("FiniteDifference/finite_differences.jl")

struct FDProblem{T<:FDTriangulation,B<:Function,L<:Function,
                V<:AbstractArray,P,BC<:Function}
    grid::T
    order::Int
    # The matrix free equivalent of the bilinear form (stiffness matrix)
    bilinearForm::B
    linearForm::L
    u::V
    parameters::P
    boundaryCondition::BC
end

function FDProblem(order,bc,operator,scheme,rhs_function,p,x,Δx)
    grid = FDTriangulation(x,Δx)
    u = zeros(length.(grid.domain)...)

    bilinearOperator = select_discretisation(operator,scheme)
    linearOperator = rhs_function
    boundaryCondition = select_bc(bc)

    return FDProblem(
        grid,
        order,
        bilinearOperator,
        linearOperator,
        u,
        p,
        boundaryCondition
    )
end


function (a::FDProblem)(dϕ, ϕ,p,t)
    apply_bilinearForm! = a.bilinearForm
    apply_linearForm!   = a.linearForm
    apply_bc!           = a.boundaryCondition
    h                   = a.grid.h
    params              = a.parameters

    apply_bilinearForm!(dϕ,ϕ,h)
    apply_linearForm!(dϕ,ϕ,params)
    apply_bc!(dϕ)
end
