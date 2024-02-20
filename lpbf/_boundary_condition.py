from dolfinx.fem import (dirichletbc, FunctionSpace, Function,
                         Constant, locate_dofs_topological)
from petsc4py.PETSc import ScalarType
import numpy as np

class BCs:
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

    def _setup_bcs(self):
        self.facet_dim = self.mesh.topology.dim - 1
        #bc_solid = self.__setup_solid_bc()
        #bc_liquid = self.__setup_liquid_bc()
        #bc_gas = self.__setup_gas_bc()
        #bc_p = self.__setup_pressure_bc()
        #bc_u = self.__setup_velocity_bc()
        bc_T = self.__setup_temperature_bc()
        #self.bcs = [*bc_solid,*bc_liquid,*bc_gas,*bc_p,*bc_u,*bc_T]
        self.bcs = [*bc_T]
    
    def __get_boundary_dofs(self, fs: FunctionSpace, marker: str) -> np.ndarray:
        # For mixed spaces, a mapping between collapsed and mixed space
        # is needed, c.f. https://fenicsproject.discourse.group/t/solving-stokes-equation-with-newtonsolver/12892/4
        return locate_dofs_topological(V=fs,
                                       entity_dim=self.facet_dim,
                                       entities=self.facet_tags.find(
                                           self.bc_markers[marker]))
    
    def __setup_solid_bc(self) -> None:
        fs = self.function_space.sub(0)
        val_inlet = ScalarType(0.0)
        inlet_dofs = self.__get_boundary_dofs(fs=fs,marker="inlet")
        bottom_dofs = self.__get_boundary_dofs(fs=fs,marker="bottom")
        bc_inlet = dirichletbc(V=fs,value=val_inlet,dofs=inlet_dofs)
        val_bottom = ScalarType(1.0)
        bc_bottom = dirichletbc(V=fs,value=val_bottom,dofs=bottom_dofs)
        return [bc_inlet,bc_bottom]
    
    def __setup_liquid_bc(self) -> None:
        fs = self.function_space.sub(1)
        inlet_dofs = self.__get_boundary_dofs(fs=fs,marker="inlet")
        bottom_dofs = self.__get_boundary_dofs(fs=fs,marker="bottom")
        val_inlet = ScalarType(0.0)
        bc_inlet = dirichletbc(V=fs,value=val_inlet,dofs=inlet_dofs)
        val_bottom = ScalarType(0.0)
        bc_bottom = dirichletbc(V=fs,value=val_bottom,dofs=bottom_dofs)
        return [bc_inlet,bc_bottom]

    def __setup_gas_bc(self) -> None:
        fs = self.function_space.sub(2)
        inlet_dofs = self.__get_boundary_dofs(fs=fs,marker="inlet")
        bottom_dofs = self.__get_boundary_dofs(fs=fs,marker="bottom")
        val_inlet = ScalarType(1.0)
        bc_inlet = dirichletbc(V=fs,value=val_inlet,dofs=inlet_dofs)
        val_bottom = ScalarType(0.0)
        bc_bottom = dirichletbc(V=fs,value=val_bottom,dofs=bottom_dofs)
        return [bc_inlet,bc_bottom]

    def __setup_pressure_bc(self) -> None:
        fs = self.function_space.sub(3)
        outlet_dofs = self.__get_boundary_dofs(fs=fs,marker="outlet")
        val = ScalarType(0.0)
        bc = dirichletbc(V=fs,value=val,dofs=outlet_dofs)
        return [bc]
    
    def __setup_velocity_bc(self) -> None:
        fs = self.function_space.sub(4)
        subspace, _ = fs.collapse()
        inlet_dofs  = locate_dofs_topological(V=(fs,subspace),
                                       entity_dim=self.facet_dim,
                                       entities=self.facet_tags.find(
                                           self.bc_markers["inlet"]))
        outlet_dofs = locate_dofs_topological(V=(fs,subspace),
                                       entity_dim=self.facet_dim,
                                       entities=self.facet_tags.find(
                                           self.bc_markers["outlet"]))
        def BC_u(x):
            dim = self.mesh.topology.dim
            values = np.zeros((dim,x.shape[1]),dtype=ScalarType)
            values[1,:] = ScalarType(1.0)
            return values
        u_bc = Function(subspace, dtype=ScalarType)
        u_bc.interpolate(BC_u)
        bc_in = dirichletbc(value=u_bc,dofs=inlet_dofs,V=fs)
        bc_out = dirichletbc(value=u_bc,dofs=outlet_dofs,V=fs)
        return [bc_in,bc_out]
    
    def __setup_temperature_bc(self) -> None:
        #fs = self.function_space.sub(5)
        fs = self.function_space
        inlet_dofs  = self.__get_boundary_dofs(fs=fs,marker="inlet")
        bottom_dofs = self.__get_boundary_dofs(fs=fs,marker="bottom")
        val_bottom = ScalarType(473.0)
        val_inlet  = ScalarType(298.0)
        bc_bottom = dirichletbc(V=fs,value=val_bottom,dofs=bottom_dofs)
        bc_inlet  = dirichletbc(V=fs,value=val_inlet,dofs=inlet_dofs)
        return [bc_bottom,bc_inlet]
