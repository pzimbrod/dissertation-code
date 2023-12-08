include("triangulation.jl")
include("FiniteElement/finite_element.jl")
include("FiniteElement/mapping.jl")
include("FiniteElement/assemble.jl")

struct AssemblyCache{T<:AbstractVecOrMat}
    coeffs::T
    loc::T
    glob::T
end

using SparseArrays

struct FEProblem{T<:Triangulation,E<:FiniteElement,J,M<:AbstractMatrix,
                V<:AbstractVector,P,F<:Function,C}
    Mesh::T
    referenceElements::E
    detJ::J
    # The matrix that gathers all discretised spatial operators
    bilinearForm::M
    # Right hand side
    linearForm::V
    u::V
    massMatrix::Union{M,Nothing}
    parameters::P
    rhs::F
    cache::C
end

# For SEM to work, we need to replace the stiffness matrix K by (M \ K).
# This in fact gives a matrix that exactly represents the central difference laplace stencil
# Also, we need to invert the mass matrix. As it is diagonal, we only need to invert M[i,i]
function preprocess_spectral!(K,Mass)
    K .= Mass \ K
    Mass .= map(x -> !iszero(x) ? inv(x) : x, Mass)
end

function FEProblem(shape::Primitive,type::ElementType,x,y,Δx,Δy,order,rhs_function!,p)

    mesh = FETriangulation(shape,x,y,Δx,Δy)
    ϕ = zeros(size(mesh.vertices,1))
    element = FiniteElement(type,shape,order)

    J_global = compute_global_jacobian(element,mesh)
    J⁻¹ = mapslices(inv,J_global,dims=(2,3))
    detJ = det.(eachslice(J_global,dims=1))
    @tullio transform[el,i,k] := J⁻¹[el,i,j] * J⁻¹[el,k,j] * detJ[el]

    K = spzeros(eltype(mesh.vertices),size(mesh.vertices,1),size(mesh.vertices,1))
    K_ref = ∫∇u∇v(element)
    assemble_global_bilinearform!(K,K_ref,mesh,transform)

    Mass = spzeros(size(K))
    M_ref = ∫uv(element)
    assemble_global_bilinearform!(Mass,M_ref,mesh,detJ)

    # For SEM, do M → M⁻¹ and K → M⁻¹ K
    type == Spectral() && preprocess_spectral!(K,Mass)

    F = similar(ϕ)
    cache = AssemblyCache(
        zeros(eltype(F),element.ndofs),
        zeros(eltype(ϕ),element.ndofs),
        zeros(eltype(F),size(F))
    )
    assemble_F!(F,cache,ϕ,element,mesh,detJ,p,rhs_function!,Mass)

    return FEProblem(
        mesh,
        element,
        detJ,
        K,
        F,
        ϕ,
        Mass,
        p,
        rhs_function!,
        cache
    )
end

function FEProblem(shape::Interval,type::ElementType,x,Δx,order,rhs_function!,p)

    mesh = FETriangulation(shape,x,Δx)
    ϕ = zeros(size(mesh.vertices,1))
    element = FiniteElement(type,shape,order)

    J_global = compute_global_jacobian(element,mesh)
    J⁻¹ = mapslices(inv,J_global,dims=(2,3))
    detJ = det.(eachslice(J_global,dims=1))
    @tullio transform[el,i,k] := J⁻¹[el,i,j] * J⁻¹[el,k,j] * detJ[el]

    K = spzeros(eltype(mesh.vertices),size(mesh.vertices,1),size(mesh.vertices,1))
    K_ref = ∫∇u∇v(element)
    assemble_global_bilinearform!(K,K_ref,mesh,transform)

    Mass = spzeros(size(K))
    M_ref = ∫uv(element)
    assemble_global_bilinearform!(Mass,M_ref,mesh,detJ)

    # For SEM, do M → M⁻¹ and K → M⁻¹ K
    type == Spectral() && preprocess_spectral!(K,Mass)

    F = similar(ϕ)
    cache = AssemblyCache(
        zeros(eltype(F),element.ndofs),
        zeros(eltype(ϕ),element.ndofs),
        zeros(eltype(F),size(F))
    )
    assemble_F!(F,cache,ϕ,element,mesh,detJ,p,rhs_function!,Mass)

    return FEProblem(
        mesh,
        element,
        detJ,
        K,
        F,
        ϕ,
        Mass,
        p,
        rhs_function!,
        cache
    )
end

function generate_ODEProblem(prob,tspan)
    type = prob.referenceElements.element_type
    # If we employ SEM, we can do away with the mass matrix
    f = type == Spectral() ? prob : ODEFunction(prob,mass_matrix=prob.massMatrix)
    ode_prob = ODEProblem(f,prob.u,tspan)
    return ode_prob
end

function (a::FEProblem)(du,u,p,t)
    mesh, element       = a.Mesh, a.referenceElements
    K, F, detJ          = a.bilinearForm, a.linearForm, a.detJ
    params,cache,rhs    = a.parameters, a.cache, a.rhs
    M                   = a.massMatrix

    assemble_F!(F,cache,u,element,mesh,detJ,params,rhs,M)
    # 5-argument mul! can use FMA instructions
    mul!(du,K,u,-1.,0.)
    du .+= F
end
