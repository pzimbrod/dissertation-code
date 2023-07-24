from numpy.linalg import norm
from dolfinx.fem import form
from dolfinx.fem import Function
from ufl import (FacetNormal, FiniteElement, Identity, Measure, TestFunction, TrialFunction, VectorElement,
                 MixedElement,as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, variable,
                 jump,avg, Form, Identity, outer, Dx)
from ufl.algebra import Power, Sum

def upwind(velocity: Function,n: FacetNormal, trial: TrialFunction) -> Form:
    vel_n = 0.5*(dot(velocity, n) + abs(dot(velocity, n)))
    return jump(vel_n * trial)

def lax_friedrichs(velocity: Function,n: FacetNormal, trial: TrialFunction) -> Form:
    v_max = max(max(velocity),0)
    return dot(avg(velocity * trial),n('+')) + 0.5 * v_max * jump(trial)

def HLLE(velocity: Function,n: FacetNormal, trial: TrialFunction) -> Form:
    v_max = max(max(velocity),0)
    v_min = min(min(velocity),0)
    return v_max/(v_max-v_min) * dot(velocity('+'),n('+'))*trial('+') - \
          v_min * dot(velocity('-'),n('+'))*trial('-') - v_max * v_min * jump(trial)

def interface_marker(alpha: Function) -> Form:
    dim = 2
    return Power(Sum(
        *[Power(Dx(alpha,i),2) for i in range(0,dim)]),
        0.5
    )

def interface_normal(alpha: Function) -> Form:
    mag_alpha = interface_marker(alpha=alpha)
    return 1/mag_alpha * grad(alpha)

def capillary_stress(alpha: Function,n: FacetNormal) -> Form:
    (dim,) = n.ufl_shape
    I = Identity(dim=dim)
    n = interface_normal(alpha=alpha)
    return interface_marker(alpha=alpha) * (I-outer(n,n))

def DG_gradient(test: TestFunction, fn: Function, u: Function, n: FacetNormal, flux_function):
    return inner(fn,div(test*u))*dx - jump(test)*flux_function(u,n,fn)*ds

def generate_weak_form(mesh,trialFunctions,testFunctions,functions,flux_function):
    test_as, test_al, test_ag, test_T, test_p, test_u = testFunctions["test_alpha_solid"], testFunctions["test_alpha_liquid"], \
                            testFunctions["test_alpha_gas"], testFunctions["test_T"], testFunctions["test_p"], testFunctions["test_u"]

    dalpha_solid, dalpha_liquid, dalpha_gas  = trialFunctions["dalpha_solid"], trialFunctions["dalpha_liquid"], \
                                                trialFunctions["dalpha_gas"]
    du, dp, dT                               = trialFunctions["du"], trialFunctions["dp"], trialFunctions["dT"]

    alpha_solid, alpha_liquid, alpha_gas  = functions["alpha_solid"], functions["alpha_liquid"], \
                                            functions["alpha_gas"]
    u, p, T                            = functions["u"], functions["p"], functions["T"]

    n = FacetNormal(mesh)

    # Mass matrices
    F = inner(test_as,dalpha_solid)*dx
    F += inner(test_al,dalpha_liquid)*dx
    F += inner(test_ag,dalpha_gas)*dx
    F += inner(test_p,dp)*dx
    F += inner(test_u,du)*dx
    F += inner(test_T,dT)*dx

    # VoF
    F += DG_gradient(test=test_as,fn=alpha_solid,u=u,n=n,flux_function=flux_function)
    F += DG_gradient(test=test_al,fn=alpha_liquid,u=u,n=n,flux_function=flux_function)
    F += DG_gradient(test=test_ag,fn=alpha_gas,u=u,n=n,flux_function=flux_function)
    # Pressure
    F += -inner(p,div(test_u))*dx
    # Continuity
    F += -inner(div(u),test_p)*dx
    # Velocity
    F += inner(test_u,div(capillary_stress(alpha=alpha_liquid,n=n)))*dx
    # Temperature

    A, l = lhs(F), rhs(F)
    return A, l
