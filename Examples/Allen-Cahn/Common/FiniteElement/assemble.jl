#=
In an n-dimensional problem setting, we have that ∇ϕ has more than one spatial components.
Thus, evaluating the weak form is more complicated.
For the FEM, we wish to pre-compute the weak forms in the reference element and transform to
the physical domain later. When gradients are involved, we cannot do this in a monolithic
step. This is because we need to multiply ∇ϕ ∇ψ with the corresponding entry of the jacobian
matrix, e.g.: ∇₁ϕᵢ ∇₂ψⱼ = ∇ϕᵢ ∇ψⱼ J⁻¹₁₂
For this reason, we store the local matrices in split form where we need (dim)² submatrices.
the output is an array K[:,:,l,m] where l and m are indices of the derivative components
=#
function ∫∇u∇v(element)
    grad_basis, weights = element.grad_basis_at_quad, element.quadrature_weights
    K_ref = zeros(eltype(weights),size(grad_basis,2),size(grad_basis,2),
            size(grad_basis,3),size(grad_basis,3))
    @tullio K_ref[i,j,l,m] = grad_basis[k,i,l] * grad_basis[k,j,m] * weights[k]
    return K_ref
end

# A very flexible macro for writing Einstein summation expressions
using Tullio

# Evaluate ∫ ψ f dx = ∫ ψ ∑ₖ fₖ ψₖ dx = ∑ⱼ ψᵢ(xⱼ) ∑ₖ fₖ ψₖ(xⱼ) wⱼ
function assemble_local_F!(F_local,element,coeffs)
    basis, weights = element.basis_at_quad, element.quadrature_weights
    @tullio F_local[l] = basis[k,l] * coeffs[j] * basis[k,j] * weights[k]
end

function assemble_F!(F,cache,ϕ,element,mesh,detJ,p,rhs_function!,massMatrix)
    connectivity    = mesh.connectivity
    coeffs          = cache.coeffs
    F_local         = cache.loc
    F_global        = cache.glob
    # Reset for accumulation
    F_global .= zero(eltype(F_global))
    @views for i in axes(connectivity,1)
        # current cell indices
        local_to_global = connectivity[i,:]
        # Coefficients of the Finite Element projection - point evaluation
        rhs_function!(coeffs,ϕ[local_to_global],p)
        # Assemble local contribution
        assemble_local_F!(F_local,element,coeffs)
        # Accumulate to global vector
        @. F_global[local_to_global] += F_local * detJ[i]
    end
    # For SEM, scale with M⁻¹ directly
    if element.element_type == Spectral()
        mul!(F,massMatrix,F_global)
    else
        F .= F_global
    end
end

function ∫uv(element)
    basis, weights = element.basis_at_quad, element.quadrature_weights
    @tullio M_ref[i,j] := basis[k,i] * basis[k,j] * weights[k]
    return M_ref
end

function ∫gv(element,apply_fun!,p)
    basis, weights = element.basis_at_quad, element.quadrature_weights
    f = similar(basis)
    apply_fun!(f,basis,p)
    @tullio M_ref[i,j] := basis[k,i] * f[k,j] * weights[k]
    return M_ref
end


function assemble_global_bilinearform!(M,M_element::AbstractMatrix,mesh,detJ)
    conn = mesh.connectivity
    M .= zero(eltype(M))
    @views for i in axes(conn,1)
        M[conn[i,:],conn[i,:]] .+= detJ[i] .* M_element
    end
end

function assemble_global_bilinearform!(K,K_element::AbstractArray{T,4},mesh,transform) where T
    conn = mesh.connectivity
    K .= zero(eltype(K))
    @tullio K[conn[g,i],conn[g,j]] += transform[g,l,m] * K_element[i,j,l,m]
    return K
end
