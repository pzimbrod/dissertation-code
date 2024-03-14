from dolfinx.fem import (FunctionSpace, locate_dofs_topological,
                         dirichletbc)
from .Mesh import Mesh
from petsc4py.PETSc import ScalarType
import numpy as np

class BoundaryConditions:
    """
    Implements the following boundary conditions:

                  | inlet           outlet          bottom          walls
    ----------------------------------------------------------------------------
    alpha_solid   | 0 Dirichlet     0 Neumann       1 Dirichlet     0 Neumann
    alpha_liquid  | 0 Dirichlet     0 Neumann       0 Dirichlet     0 Neumann
    alpha_gas     | 1 Dirichlet     0 Neumann       0 Dirichlet     0 Neumann
    p             | 0 Neumann       0 Dirichlet     0 Neumann       0 Neumann
    u             | 1 Dirichlet     1 Dirichlet     0 Neumann       0 Neumann
    T             | 298 Dirichlet   0 Neumann       473 Dirichlet   0 Neumann
    """

    def __init__(self, mesh: Mesh, 
                 function_spaces: dict[str,FunctionSpace]) -> None:
        self.T = self._setup_temperature_bc(mesh=mesh,
                                            fs=function_spaces["T"])
        
        return
    

    def _get_boundary_dofs(self, fs: FunctionSpace,
                           mesh: Mesh, marker: str) -> np.ndarray:
        dofs = locate_dofs_topological(V=fs,
                                       entity_dim=mesh.facet_dim,
                                       entities=mesh.facet_tags.find(
                                           mesh.bc_markers[marker]))
        
        return dofs
    

    def _setup_temperature_bc(self, mesh: Mesh, fs: FunctionSpace) -> None:
        inlet_dofs  = self._get_boundary_dofs(fs=fs,mesh=mesh,
                                               marker="inlet")
        bottom_dofs = self._get_boundary_dofs(fs=fs,mesh=mesh,
                                               marker="bottom")

        val_bottom = ScalarType(473.0)
        val_inlet  = ScalarType(298.0)

        bc_bottom = dirichletbc(V=fs,value=val_bottom,dofs=bottom_dofs)
        bc_inlet  = dirichletbc(V=fs,value=val_inlet,dofs=inlet_dofs)

        return [bc_bottom,bc_inlet]