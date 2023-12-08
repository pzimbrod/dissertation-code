using LinearAlgebra, FastGaussQuadrature, Tullio

abstract type ElementType end
struct Lagrange <: ElementType end
struct Spectral <: ElementType end

struct FiniteElement{E<:ElementType,P<:Primitive,B<:AbstractMatrix,Q<:AbstractVector,G<:AbstractArray}
    primitive::P
    element_type::E
    order::Int
    ndofs::Int
    basis_coeffs::B
    quadrature_nodes::B
    quadrature_weights::Q
    basis_at_quad::B
    grad_basis_at_quad::G
    grad_monomial_basis::G
end

function FiniteElement(type,primitive,order;quadrature_alg=gausslegendre,compute_gradient=true)
    dim = primitive_dim(primitive)
    x_ref = reference_domain(type,dim,order)
    alg = type == Spectral() ? gausslobatto : quadrature_alg
    nodes, weights = quadrature_nodes_weights(primitive,alg,order)
    type != Spectral() && transform_reference_coordinates!(nodes,weights)

    # Create the nodal basis, i.e. the monomial basis at the element nodes
    monomial_basis = create_monomial_basis(x_ref,order)
    basis_coeffs = monomial_basis \ I
    ndofs = size(basis_coeffs,1)

    # Evaluate reference basis at quadrature points
    # The resulting matrix (i,j) has dim i = number of quadrature points and
    # j = number of DoFs (i.e. shape functions)
    basis = tabulate(nodes,order,basis_coeffs)
    grad_basis = compute_gradient ? grad_tabulate(nodes, order, basis_coeffs) : nothing
    monomial_grad = grad_tabulate(x_ref,order,basis_coeffs)
    return FiniteElement(
        primitive,
        type,
        order,
        ndofs,
        basis_coeffs,
        nodes,
        weights,
        basis,
        grad_basis,
        monomial_grad
    )
end

function create_monomial_basis(x_ref,order)
    dim = size(x_ref,2)
    # This is only valid for tensor product primitives (Interval, Quadrilateral, Hexahedron)
    ndofs = Int((order+1)^dim)
    vandermonde = zeros(size(x_ref,1),ndofs)
    _fill_vandermonde_matrix!(Val(dim),vandermonde,x_ref,order)
    return vandermonde
end

function _fill_vandermonde_matrix!(::Val{1},vandermonde,vertices,order)
    idx = 1
    @views for i in 0:order
        @. vandermonde[:,idx] += vertices[:,1]^i
        idx += 1
    end
end
function _fill_gradient_vandermonde_matrix!(::Val{1},dvandermonde,vertices,order)
    idx = 2
    @views for i in 0:order
        # Constant term vanishes
        i == 0 && continue
        # negative exponents give NaN
        @. dvandermonde[:,idx,1] = i * vertices[:,1]^max(i-1,0)
        idx += 1
    end
end

function _fill_vandermonde_matrix!(::Val{2},vandermonde,vertices,order)
    idx = 1
    @views for i in 0:order, j in 0:order
        @. vandermonde[:,idx] += vertices[:,1]^i * vertices[:,2]^j
        idx += 1
    end
end
function _fill_gradient_vandermonde_matrix!(::Val{2},dvandermonde,vertices,order)
    idx = 2
    @views for i in 0:order, j in 0:order
        # Constant term vanishes
        (i == 0 && j == 0) && continue
        # negative exponents give NaN
        @. dvandermonde[:,idx,1] = i * vertices[:,1]^max(i-1,0) * vertices[:,2]^j
        @. dvandermonde[:,idx,2] = vertices[:,1]^i * j * vertices[:,2]^max(j-1,0)
        idx += 1
    end
end

#= Tabulation is the process of evaluating the basis functions ϕⱼ at an arbitrary set
of points. This is needed for numerical quadrature. Then, we need the basis functions
evaluated at the quadrature points. =#
function tabulate(nodes, order, basis_coeffs)
    vandermonde = create_monomial_basis(nodes,order)
    return vandermonde * basis_coeffs
end

function grad_monomial_basis(x_ref,order)
    dim = size(x_ref,2)
    # This is only valid for tensor product primitives (Interval, Quadrilateral, Hexahedron)
    ndofs = Int((order+1)^dim)
    # The gradient vandermonde array has a third dim to acommodate the spatial components
    # of the gradient
    dvandermonde = zeros(size(x_ref,1),ndofs,dim)
    _fill_gradient_vandermonde_matrix!(Val(dim),dvandermonde,x_ref,order)

    return dvandermonde
end

function grad_tabulate(nodes, order, basis_coeffs)
    dvandermonde = grad_monomial_basis(nodes,order)
    @tullio out[i,j,k] := dvandermonde[i,l,k] * basis_coeffs[l,j]
    return out
end

using FastGaussQuadrature
# Transform from [-1;1] to [0;1]
function transform_reference_coordinates!(nodes,weights)
    @. nodes = 0.5 * nodes + 0.5
    @. weights = 0.5 * weights
    return nothing
end

function quadrature_nodes_weights(primitive::TensorProductPrimitive,alg::Function,degree)
    dim = primitive_dim(primitive)

    nodes, weights = alg(degree+1)
    dim == 1 && return nodes[:,:], weights
    n = length(nodes)

    nD_nodes, nD_weights = _extend(Val(dim),n,nodes), _extend(Val(dim),n,weights)

    nD_weights_vec = reduce(*,nD_weights, dims=2) |> vec

    return nD_nodes, nD_weights_vec
end

# Extending 1D nodes to 1D does nothing
_extend(::Val{1},n,x) = x

function _extend(::Val{2},n,x)
    x_2D = reduce(hcat,[
        repeat(x,outer=n),
        repeat(x, inner=n)
    ])
    return x_2D
end

function _extend(::Val{3},n,x)
    x_2D = _extend(Val(2),n,x)
    x_3D = reduce(hcat,[
        repeat(x_2D,outer=(n,1)),
        repeat(x, inner=n*n)
    ])
    return x_3D
end


function reference_domain(eltype,dim,order)
    domain_1D = reference_1d_domain(eltype,order)
    domain = _extend(Val(dim),length(domain_1D),domain_1D)
    return domain
end

reference_1d_domain(::Lagrange,order) = range(0.,1.,length=order+1)
reference_1d_domain(::Spectral,order) = range(-1.,1.,length=order+1)
