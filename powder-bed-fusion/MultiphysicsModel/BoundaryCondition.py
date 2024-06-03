from dolfinx.fem import (FunctionSpace, locate_dofs_topological,
                         dirichletbc, Function)
from .Mesh import AbstractMesh
from .FEData import AbstractFEData
from petsc4py.PETSc import ScalarType
import numpy as np

class AbstractBoundaryConditions:
    def __init__(self) -> None:
        self.functions = {}

        return


    def apply(self, mesh: AbstractMesh, fe_data:AbstractFEData) -> None:
        """
        for each function present in `fe_data`, interpolate the
        respective BC
        
        Parameters
        ----------

        `fe_data` : `FEData`
            the data structure containing Finite Elemenent specific
            attributes of the problem
        
        `mesh` : `Mesh`
            the computational mesh of the problem
        """
        self.bc_list = []
        for (field,bc_fun) in self.functions.items():
            if field in fe_data.config.keys():
                self.bc_list.extend(bc_fun(mesh=mesh,fe_data=fe_data))
        
        return




class RBBoundaryConditions(AbstractBoundaryConditions):
    def __init__(self) -> None:
        super().__init__()

        return
    





class PBFBoundaryConditions(AbstractBoundaryConditions):
    """
    Implements the following boundary conditions:

    =============== ================= ================= ================= ===============
                    inlet             outlet            bottom            walls
    =============== ================= ================= ================= ===============
    alpha_solid     Dirichlet 0       Neumann   0       Dirichlet 1       Neumann 0
    alpha_liquid    Dirichlet 0       Neumann   0       Dirichlet 0       Neumann 0
    alpha_gas       Dirichlet 1       Neumann   0       Dirichlet 0       Neumann 0
    p               Neumann   0       Dirichlet 0       Neumann   0       Neumann 0
    u               Dirichlet (0,1,0) Dirichlet (0,1,0) Neumann   (0,0,0) Neumann (0,0,0)
    T               Dirichlet 298     Neumann   0       Dirichlet 473     Neumann 0
    =============== ================= ================= ================= ===============

    Attributes
    ----------
    `functions` : `dict[str,function]`
        for each field, contains the function that interpolates
        the respective boundary condition


    Methods
    -------
    apply(fe_data:AbstractFEData)
        for each function present in `fe_data`, interpolate the
        respective BC
    """

    def __init__(self) -> None:
        super().__init__()
        self.functions["alpha_solid"]  =  self._alpha_solid
        self.functions["alpha_liquid"] = self._alpha_liquid
        self.functions["alpha_gas"]    = self._alpha_gas
        self.functions["p"]            = self._p
        self.functions["u"]            = self._u
        self.functions["T"]            = self._T
        
        return
    

    def _get_boundary_dofs(self, fs: FunctionSpace,
                           mesh: AbstractMesh, marker: str,
                           mixed_space: bool = False) -> np.ndarray:
        if mixed_space:
            subspace, _ = fs.collapse()
            dofs = locate_dofs_topological(V=(fs,subspace),
                                       entity_dim=mesh.facet_dim,
                                       entities=mesh.facet_tags.find(
                                           mesh.bc_markers[marker]))
        else:
            dofs = locate_dofs_topological(V=fs,
                                        entity_dim=mesh.facet_dim,
                                        entities=mesh.facet_tags.find(
                                            mesh.bc_markers[marker]))
        
        return dofs
    

    def _alpha_solid(self, fe_data:AbstractFEData, mesh: AbstractMesh) -> None:
        fs=fe_data.function_spaces["alpha_solid"]
        val_inlet = ScalarType(0.0)
        inlet_dofs = self._get_boundary_dofs(fs=fs,mesh=mesh,
                                             marker="inlet")
        bc_inlet = dirichletbc(V=fs,value=val_inlet,dofs=inlet_dofs)

        bottom_dofs = self._get_boundary_dofs(fs=fs,mesh=mesh,
                                              marker="bottom")
        val_bottom = ScalarType(1.0)
        bc_bottom = dirichletbc(V=fs,value=val_bottom,dofs=bottom_dofs)

        return [bc_inlet,bc_bottom]

    
    def _alpha_liquid(self, fe_data:AbstractFEData, mesh: AbstractMesh) -> None:
        fs=fe_data.function_spaces["alpha_liquid"]
        inlet_dofs = self._get_boundary_dofs(fs=fs,mesh=mesh,
                                             marker="inlet")
        val_inlet = ScalarType(0.0)
        bc_inlet = dirichletbc(V=fs,value=val_inlet,dofs=inlet_dofs)

        bottom_dofs = self._get_boundary_dofs(fs=fs,mesh=mesh,
                                              marker="bottom")
        val_bottom = ScalarType(0.0)
        bc_bottom = dirichletbc(V=fs,value=val_bottom,dofs=bottom_dofs)

        return [bc_inlet,bc_bottom]


    def _alpha_gas(self, fe_data:AbstractFEData, mesh: AbstractMesh) -> None:
        fs=fe_data.function_spaces["alpha_gas"]
        inlet_dofs = self._get_boundary_dofs(fs=fs,mesh=mesh,
                                             marker="inlet")
        val_inlet = ScalarType(1.0)
        bc_inlet = dirichletbc(V=fs,value=val_inlet,dofs=inlet_dofs)

        bottom_dofs = self._get_boundary_dofs(fs=fs,mesh=mesh,
                                              marker="bottom")
        val_bottom = ScalarType(0.0)
        bc_bottom = dirichletbc(V=fs,value=val_bottom,dofs=bottom_dofs)

        return [bc_inlet,bc_bottom]


    def _p(self, fe_data:AbstractFEData, mesh: AbstractMesh) -> None:
        fs=fe_data.function_spaces["p"]
        outlet_dofs = self._get_boundary_dofs(fs=fs,mesh=mesh,
                                              marker="outlet")
        val = ScalarType(0.0)
        bc_outlet = dirichletbc(V=fs,value=val,dofs=outlet_dofs)

        return [bc_outlet]

    
    def _u(self, fe_data:AbstractFEData, mesh: AbstractMesh) -> None:
        fs=fe_data.function_spaces["u"]
        is_mixed = fe_data.is_mixed
        dim = mesh.cell_dim
        inlet_dofs   = self._get_boundary_dofs(fs=fs,mesh=mesh,
                                              marker="inlet",
                                              mixed_space=is_mixed)

        outlet_dofs  = self._get_boundary_dofs(fs=fs,mesh=mesh,
                                              marker="outlet",
                                              mixed_space=is_mixed)
        def BC_u(x):
            values = np.zeros((dim,x.shape[1]),dtype=ScalarType)
            values[1,:] = ScalarType(1.0)
            return values
        
        if fe_data.is_mixed:
            space, _ = fs.collapse()
            u_bc = Function(space, dtype=ScalarType)
            u_bc.interpolate(BC_u)
        else:
            u_bc = np.array((0.,1.,0.), dtype=ScalarType)


        bc_in = dirichletbc(value=u_bc,dofs=inlet_dofs,V=fs)
        bc_out = dirichletbc(value=u_bc,dofs=outlet_dofs,V=fs)

        return [bc_in,bc_out]


    def _T(self, fe_data:AbstractFEData, mesh: AbstractMesh) -> None:
        fs=fe_data.function_spaces["T"]

        inlet_dofs  = self._get_boundary_dofs(fs=fs,mesh=mesh,
                                               marker="inlet")
        val_inlet  = ScalarType(298.0)
        bc_inlet  = dirichletbc(V=fs,value=val_inlet,dofs=inlet_dofs)

        bottom_dofs = self._get_boundary_dofs(fs=fs,mesh=mesh,
                                               marker="bottom")
        val_bottom = ScalarType(473.0)
        bc_bottom = dirichletbc(V=fs,value=val_bottom,dofs=bottom_dofs)

        return [bc_bottom,bc_inlet]