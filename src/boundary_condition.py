from dolfinx.fem import dirichletbc, locate_dofs_topological, Constant, Function
from petsc4py import PETSc
import numpy as np

"""
              | inlet           outlet          bottom          walls
----------------------------------------------------------------------------
alpha_solid   | 0 Neumann       0 Neumann       0 Neumann       0 Neumann
alpha_liquid  | 0 Neumann       0 Neumann       0 Neumann       0 Neumann
alpha_gas     | 0 Neumann       0 Neumann       0 Neumann       0 Neumann
u             | 1 Dirichlet     0 Neumann       0 Neumann       0 Neumann
p             | 0 Neumann       0 Dirichlet     0 Neumann       0 Neumann
T             | 0 Neumann       0 Neumann       Dirichlet 473   0 Neumann
"""
def define_boundary_conditions(mesh,facets,markers,function_spaces):
    fdim = mesh.topology.dim - 1
    inlet_marker, outlet_marker, wall_marker, bottom_marker = markers
    fs_cg_scalar, fs_dg_scalar, fs_cg_vector = function_spaces

    # u
    u_inlet = Function(fs_cg_vector)
    inlet_vel = np.array([0.,1.,0.], dtype=PETSc.ScalarType)
    u_inlet.interpolate(inlet_vel)
    bc_u_inlet = dirichletbc(u_inlet, locate_dofs_topological(fs_cg_vector, fdim, 
                                                              facets.find(inlet_marker)))
    
    # p
    p_outlet = Constant(mesh, PETSc.ScalarType(0.))
    bc_p_outlet = dirichletbc(p_outlet, locate_dofs_topological(fs_dg_scalar, fdim, 
                                                              facets.find(outlet_marker)))

    # T
    T_bottom = Constant(mesh, PETSc.ScalarValue(473.))
    bc_T_bottom = dirichletbc(T_bottom, locate_dofs_topological(fs_cg_scalar, fdim, 
                                                              facets.find(bottom_marker)))

    bcs = {
        "u":            bc_u_inlet,
        "p":            bc_p_outlet,
        "T":            bc_T_bottom,
        "alpha_solid":  [],
        "alpha_liquid": [],
        "alpha_gas":    []
    }

    return bcs