from dolfinx.fem import Constant
from petsc4py import PETSc
from ufl import CellDiameter

def setup_constants(mesh):
    #### Constants ###
    rho = Constant(mesh,PETSc.ScalarType(7800))      # Density [kg/m^3]
    eta = Constant(mesh,PETSc.ScalarType(0.1))       # Viscosity [Pa s]
    kappa = Constant(mesh,PETSc.ScalarType(15.0))     # Thermal conductivity [W/(m K)]
    R = Constant(mesh,PETSc.ScalarType(7.814))
    Tv = Constant(mesh,PETSc.ScalarType(2600.))
    Hv = Constant(mesh,PETSc.ScalarType(2e5))
    p0 = Constant(mesh,PETSc.ScalarType(1.013e5))
    M = Constant(mesh,PETSc.ScalarType(4.0))
    T_amb = Constant(mesh,PETSc.ScalarType(298.0))
    epsilon = Constant(mesh,PETSc.ScalarType(0.4))  # Emissivity
    sb = Constant(mesh,PETSc.ScalarType(5.6704e-8)) # Stefan-Boltzmann constant [W m^-2 K^-4]
    c_p = Constant(mesh,PETSc.ScalarType(500.0))    # heat capacity [J / (kg K)]

    h = CellDiameter(mesh)

    #### Variables ####
    def sigma(T):
        return 1.44 - T * 2e-4
    
    constants = {
        "rho":      rho,
        "eta":      eta,
        "kappa":    kappa,
        "sigma":    sigma,
        "R":        R,
        "Tv":       Tv,
        "Hv":       Hv,
        "p0":       p0,
        "M":        M,
        "T_amb":    T_amb,
        "epsilon":  epsilon,
        "sb":       sb,
        "c_p":      c_p,
        "h":        h,
    }
    return constants