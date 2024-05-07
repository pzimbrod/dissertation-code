function compute_local_jacobian(monomial_grad,vertices)
    # The jacobian in each point is a linear combination of the shape functions in
    # physical coordinates
    # For an affine mapping, we choose X to be at the origin, since the jacobian is Constant
    # within each element
    # Jᵦᵧ(X) = ∑ⱼ (xⱼ)ᵦ ∇ᵧψⱼ(X)
    β = size(vertices,2)
    γ = size(monomial_grad,3)
    J = zeros(β,γ)
    # We simply evaluate at ξ=0 as the jacobian is constant anyway
    @tullio J[β,γ] = vertices[j,β] * monomial_grad[1,j,γ]
    return J
end

function compute_global_jacobian(element,mesh)
    monomial_grad = element.grad_monomial_basis
    vertices, connectivity = mesh.vertices, mesh.connectivity
    J_global = zeros(size(connectivity,1),size(vertices,2),size(monomial_grad,3))
    @views for i in axes(J_global,1)
        J_global[i,:,:] = compute_local_jacobian(monomial_grad,
                                                    vertices[connectivity[i,:],:])
    end
    return J_global
end
