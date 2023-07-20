from dolfinx.fem import form
from ufl import (FacetNormal, FiniteElement, Identity, Measure, TestFunction, TrialFunction, VectorElement,
                 MixedElement,as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, variable)

def generate_weak_form(mesh,trialFunctions,testFunctions,functions):
    
    v,w,q = testFunctions
    dalpha_solid, dalpha_liquid, dalpha_gas, dp, du, dT = trialFunctions
    alpha_solid, alpha_liquid, alpha_gas, p, u, T = functions

    n = FacetNormal(mesh)

    # Time dependent contributions
    F = inner(w,dalpha_solid)*dx
    F += inner(w,dalpha_liquid)*dx
    F += inner(w,dalpha_gas)*dx
    F += inner(w,dp)*dx
    F += inner(v,dT)*dx

    a, l = lhs(F), rhs(F)
    return a, l
