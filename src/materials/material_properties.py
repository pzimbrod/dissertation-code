from dolfinx.fem import Constant
from petsc4py import PETSc

def setup_constants(mesh):
    #### Constants ###
    rho = Constant(mesh,PETSc.ScalarType(7800))      # Density [kg/m^3]
    eta = Constant(mesh,PETSc.ScalarType(0.1))       # Viscosity [Pa s]
    kappa = Constant(mesh,PETSc.ScalarType(0.1))     # Thermal conductivity [W/(m K)]

    #### Variables ####
    def sigma(T):
        return 1.44 - T * 2e-4
    
    constants = {
        "rho":      rho,
        "eta":      eta,
        "kappa":    kappa,
        "sigma":    sigma
    }
    return constants