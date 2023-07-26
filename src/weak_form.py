from dolfinx.fem import Function
from ufl import FacetNormal, TestFunction,div,nabla_div, dS, dx, inner, lhs, grad, rhs,jump, SpatialCoordinate, Form

from numerics.fluxes import HLLE, lax_friedrichs, upwind
from numerics.multiphase import interface_marker, interface_normal, capillary_stress
from materials.material_models import recoil_pressure, heat_of_vaporisation, radiation

def DG_gradient(test: TestFunction, fn: Function, u: Function, n: FacetNormal, flux_function) -> Form:
    return inner(fn,div(test*u))*dx - jump(test)*flux_function(u,n,fn)*dS

def generate_weak_form(mesh,constants,trialFunctions,testFunctions,functions,flux_function) -> Form:
    test_as, test_al, test_ag, test_T, test_p, test_u = testFunctions["test_alpha_solid"], testFunctions["test_alpha_liquid"], \
                            testFunctions["test_alpha_gas"], testFunctions["test_T"], testFunctions["test_p"], testFunctions["test_u"]

    dalpha_solid, dalpha_liquid, dalpha_gas = trialFunctions["dalpha_solid"], trialFunctions["dalpha_liquid"], \
                                                trialFunctions["dalpha_gas"]
    du, dp, dT                              = trialFunctions["du"], trialFunctions["dp"], trialFunctions["dT"]

    alpha_solid, alpha_liquid, alpha_gas    = functions["alpha_solid"], functions["alpha_liquid"], \
                                                functions["alpha_gas"]
    u, p, T                                 = functions["u"], functions["p"], functions["T"]
    kappa, rho, c_p                         = constants["kappa"], constants["rho"], constants["c_p"]

    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)

    # Mass matrices
    # Nonlinear problems do not have a TrialFunction. Instead, all unknowns are of type Function
    F = inner(test_as,alpha_solid)*dx
    F += inner(test_al,alpha_liquid)*dx
    F += inner(test_ag,alpha_gas)*dx
    F += inner(test_p,p)*dx
    F += inner(test_u,u)*dx
    F += inner(test_T,T)*rho*c_p*dx

    # VoF
    F += DG_gradient(test=test_as,fn=alpha_solid,u=u,n=n,flux_function=flux_function)
    F += DG_gradient(test=test_al,fn=alpha_liquid,u=u,n=n,flux_function=flux_function)
    F += DG_gradient(test=test_ag,fn=alpha_gas,u=u,n=n,flux_function=flux_function)
    # Pressure
    F += -inner(p,div(test_u))*dx
    F += inner(test_p,recoil_pressure(constants=constants, T=T))*dx
    # Continuity
    F += -inner(div(u),test_p)*dx
    # Velocity
    F -= inner(grad(test_u),grad(u))*dx                                    # Viscous term (viscosity missing)
    # This produces out of bounds error
    F += inner(grad(test_u),capillary_stress(alpha1=alpha_liquid,
                                           alpha2=alpha_gas,
                                           T=T,
                                           constants=constants))*dx     # Capillary Stress (ST missing)
    # Temperature
    F += kappa*inner(grad(test_T),grad(T))*dx                            # Heat conduction
    F -= inner(test_T,radiation(constants=constants, T=T))*dx
    F -= inner(test_T,heat_of_vaporisation(constants=constants,T=T,recoil_pressure=recoil_pressure))*dx

    # This will be in total a nonlinear problem, thus we return the unsplit form
    return F 
