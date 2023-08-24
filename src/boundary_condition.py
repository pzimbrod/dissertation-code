from dolfinx.fem import dirichletbc, locate_dofs_topological, Constant, Function, FunctionSpace
from dolfinx.mesh import Mesh
from petsc4py.PETSc import ScalarType
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
def define_boundary_conditions(mesh: Mesh,facets,markers,fs: FunctionSpace, functions: Function):
    fdim = mesh.topology.dim - 1
    inlet_marker, outlet_marker, wall_marker, bottom_marker = markers

    # u
    fs_u, _ = fs.sub(5).collapse()
    u_inlet = Function(fs_u)
    # Set all entries to 0
    u_inlet.x.array[:] = 0.
    # Make the velocity vector (0,1,0), i.e. set y to 1
    u_inlet.x.array[:][1] = 1.
    inlet_dofs = locate_dofs_topological((fs.sub(5),fs_u), fdim, facets.find(inlet_marker))
    bc_u_inlet = dirichletbc(u_inlet,inlet_dofs,fs.sub(5))
    
    # p
    p_outlet = Constant(mesh, ScalarType(0.))
    fs_p, _ = fs.sub(1).collapse()
    outlet_dofs = locate_dofs_topological(fs_p, fdim, facets.find(outlet_marker))
    bc_p_outlet = dirichletbc(p_outlet, outlet_dofs,fs.sub(1))

    # T
    T_bottom = Constant(mesh, ScalarType(473.))
    fs_T, _ = fs.sub(0).collapse()
    bottom_dofs = locate_dofs_topological(fs_T, fdim, facets.find(bottom_marker))
    bc_T_bottom = dirichletbc(T_bottom, bottom_dofs, fs.sub(0))

    bcs = [
        bc_u_inlet,
        bc_p_outlet,
        bc_T_bottom,
    ]

    return bcs