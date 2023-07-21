from dolfinx.fem import Constant, Function, FunctionSpace
from ufl import FiniteElement, VectorElement, TrialFunction, TestFunction
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

    fs_cg_scalar = FunctionSpace(mesh=mesh,element=fe_cg_scalar)
    fs_dg_scalar = FunctionSpace(mesh=mesh,element=fe_dg_scalar)
    fs_cg_vector = FunctionSpace(mesh=mesh,element=fe_cg_vector)
    function_spaces = [fs_cg_scalar, fs_dg_scalar, fe_cg_vector]

    # Test Functions
    v = TestFunction(fs_cg_scalar)
    w = TestFunction(fs_dg_scalar)
    q = TestFunction(fs_cg_vector)
    testFunctions = [v,w,q]

    # Time dependent problems have the time derivatives as Trial functions
    dalpha_solid = TrialFunction(fs_dg_scalar)
    dalpha_liquid = TrialFunction(fs_dg_scalar)
    dalpha_gas = TrialFunction(fs_dg_scalar)
    dp = TrialFunction(fs_dg_scalar)
    du = TrialFunction(fs_cg_vector)
    dT = TrialFunction(fs_cg_scalar)
    trialFunctions = [dalpha_solid, dalpha_liquid, dalpha_gas, dp, du, dT]

    # And the primary variables as regular Functions
    alpha_solid = Function(fs_dg_scalar)
    alpha_liquid = Function(fs_dg_scalar)
    alpha_gas = Function(fs_dg_scalar)
    p = Function(fs_dg_scalar)
    u = Function(fs_cg_vector)
    T = Function(fs_cg_scalar)
    functions = [alpha_solid, alpha_liquid, alpha_gas, p, u, T]

    return testFunctions, trialFunctions, functions, function_spaces