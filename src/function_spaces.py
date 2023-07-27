from dolfinx.fem import Constant, Function, FunctionSpace
from ufl import FiniteElement, VectorElement, TrialFunctions, TestFunctions, MixedElement
from dolfinx.mesh import Mesh

"""
Independent variables in this model:

- alpha_solid:  Solid phase fraction    (DG)
- alpha_liquid: Liquid phase fraction   (DG)
- alpha_gas:    Gaseous phase fraction  (DG)
- u (x,y,z):    Velocity                (CG)
- p:            Pressure                (DG)
- T:            Temperature             (CG)
"""
def create_fe_functions(mesh: Mesh,degree: int):
    fe_cg_scalar = FiniteElement(family="CG",cell=mesh.ufl_cell(),degree=degree)
    fe_dg_scalar = FiniteElement(family="DG",cell=mesh.ufl_cell(),degree=degree)
    fe_cg_vector = VectorElement(family="CG",cell=mesh.ufl_cell(),degree=degree,dim=3)

    # Create a mixed function space for all primary variables
    fe = MixedElement([fe_cg_scalar,fe_dg_scalar,fe_dg_scalar,fe_dg_scalar,fe_dg_scalar,fe_cg_vector])
    fs = FunctionSpace(mesh=mesh,element=fe)

    # Test Functions
    s, q, a_s, a_l, a_g, v = TestFunctions(fs)
    testFunctions = {
        "test_T":           s,
        "test_p":           q,
        "test_alpha_solid": a_s,
        "test_alpha_liquid":a_l,
        "test_alpha_gas":   a_g,
        "test_u":           v     
    }

    # Time dependent problems have the time derivatives as Trial functions
    dT, dp, dalpha_solid, dalpha_liquid, dalpha_gas, du = TrialFunctions(fs)
    trialFunctions = {
        "dalpha_solid":     dalpha_solid,
        "dalpha_liquid":    dalpha_liquid,
        "dalpha_gas":       dalpha_gas,
        "dp":               dp,
        "du":               du,
        "dT":               dT
    }

    # And the primary variables as regular Functions
    f = Function(fs)
    T, p, alpha_solid, alpha_liquid, alpha_gas, u = f.split()
    functions = {
        "alpha_solid":     alpha_solid,
        "alpha_liquid":    alpha_liquid,
        "alpha_gas":       alpha_gas,
        "p":               p,
        "u":               u,
        "T":               T
    }

    return fs, f