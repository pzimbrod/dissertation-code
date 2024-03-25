abstract type DifferentialOperator end
struct Laplacian    <: DifferentialOperator end
struct Divergenve   <: DifferentialOperator end
struct Gradient     <: DifferentialOperator end

abstract type DifferencingScheme end
struct ForwardDifference    <: DifferencingScheme end
struct BackwardDifference   <: DifferencingScheme end
struct CentralDifference    <: DifferencingScheme end

function laplacian!(dϕ::AbstractMatrix,ϕ::AbstractMatrix,h)
    Δx = reduce(*,h)
    @inbounds for i in axes(dϕ,1)[2:end-1], k in axes(dϕ,2)[2:end-1]
        dϕ[i,k] = ϕ[i+1,k] + ϕ[i-1,k] + ϕ[i,k+1] + ϕ[i,k-1] - 4*ϕ[i,k]
    end
    dϕ ./= Δx
end

function laplacian!(dϕ::AbstractVector,ϕ::AbstractVector,Δx)
    @views begin
        @. dϕ[2:end-1] = 1/Δx^2 * (ϕ[1:end-2] - 2*ϕ[2:end-1] + ϕ[3:end])
    end
end

select_discretisation(::Laplacian,::CentralDifference) = laplacian!
