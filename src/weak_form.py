from dolfinx.fem import form
from dolfinx.fem import Function
from ufl import (FacetNormal, FiniteElement, Identity, Measure, TestFunction, TrialFunction, VectorElement,
                 MixedElement,as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, variable,
                 jump,avg, Form)

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
    F += inner(test_T,dT)*dx


    A, l = lhs(F), rhs(F)
    return A, l
