from dolfinx.fem import (FunctionSpace, locate_dofs_topological,
                         dirichletbc, Function, Constant,
                         locate_dofs_geometrical)
from .Mesh import AbstractMesh
from .FEData import AbstractFEData
from petsc4py.PETSc import ScalarType
import numpy as np
from numpy.typing import ArrayLike

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
    def __init__(self, fe_data: AbstractFEData, parameters: dict[str,any]) -> None:
        super().__init__()

        self.parameters = parameters

        a = self.parameters["a"]
        self.markers = {
            "leftright":    lambda x: np.isclose(x[0], -2*a) | np.isclose(x[0],2*a),
            "top":          lambda x: np.isclose(x[1], 2*a),
            "bottom":       lambda x: np.isclose(x[1], -2*a),
        }

        self.boundary_dofs = self.__get_boundary_dofs(fe_data=fe_data)

        self.functions["alpha1"]    = self._alpha1
        self.functions["alpha2"]    = self._alpha2
        self.functions["p"]         = self._p
        self.functions["u"]         = self._u
        self.functions["T"]         = self._T

        return
    

    def __get_boundary_dofs(self, fe_data: AbstractFEData) -> dict[str,ArrayLike]:
        boundary_dofs = {}
        fs_idxs = fe_data.sub_map
        fs = fe_data.mixed_function_space
        markers = self.markers

        for (field,idx) in fs_idxs.items():
            boundary_dofs[field] = {}
            sub = fs.sub(idx)
            space, _ = sub.collapse()
            for (facet,marker) in markers.items():
                boundary_dofs[field][facet] = locate_dofs_geometrical(
                    V=(sub,space),
                    marker=marker
                )

        boundary_dofs["u_x"] = {}
        sub = fs.sub(fs_idxs["u"]).sub(0)
        space, _ = sub.collapse()
        for (facet,marker) in markers.items():
                boundary_dofs["u_x"][facet] = locate_dofs_geometrical(
                    V=(sub,space),
                    marker=marker
        )
                
        return boundary_dofs
    

    def _alpha1(self, fe_data:AbstractFEData, mesh: AbstractMesh) -> None:
        
        return []


    def _alpha2(self, fe_data:AbstractFEData, mesh: AbstractMesh) -> None:
        
        return []


    def _p(self, fe_data:AbstractFEData, mesh: AbstractMesh) -> None:
        fs=fe_data.function_spaces["p"]
        wall_dofs = self.boundary_dofs["p"]["leftright"]
        space, _ = fs.collapse()

        p_bc = Function(space)
        p_bc.interpolate(lambda x: np.full(x.shape[0],0.0))
        bc_leftright = dirichletbc(V=fs,value=p_bc,dofs=wall_dofs)

        return [bc_leftright]

    
    def _u(self, fe_data:AbstractFEData, mesh: AbstractMesh) -> None:
        fs=fe_data.function_spaces["u"]
        
        top_dofs    = self.boundary_dofs["u_x"]["top"]
        bottom_dofs = self.boundary_dofs["u_x"]["bottom"]
        wall_dofs   = self.boundary_dofs["u_x"]["leftright"]
        
        space, _ = fs.sub(0).collapse()
        # In this particular case, on all boundaries the x-component
        # of velocity is constrained to be zero
        u_x_bc = Function(space, dtype=ScalarType)
        u_x_bc.interpolate(lambda x: np.full(x.shape[0], (0.)))

        # Symmetry BC: no penetration through wall, tangential slip
        bc_leftright    = dirichletbc(value=u_x_bc, dofs=wall_dofs,V=fs.sub(0))
        # No slip
        bc_top          = dirichletbc(value=u_x_bc, dofs=top_dofs,V=fs.sub(0))
        # No slip
        bc_bottom       = dirichletbc(value=u_x_bc, dofs=bottom_dofs,V=fs.sub(0))

        return [bc_top,bc_bottom,bc_leftright]


    def _T(self, fe_data:AbstractFEData, mesh: AbstractMesh) -> None:
        fs=fe_data.function_spaces["T"]

        top_dofs = self.boundary_dofs["T"]["top"]
        space, _ = fs.collapse()
        T_top = Function(space)
        T_top.interpolate(lambda x: np.full(x.shape[0],self.parameters["temp_top"]))
        bc_top   = dirichletbc(V=fs,value=T_top,dofs=top_dofs)

        bottom_dofs = self.boundary_dofs["T"]["bottom"]
        T_bottom = Function(space)
        T_bottom.interpolate(lambda x: np.full(x.shape[0],self.parameters["temp_bottom"]))
        bc_bottom = dirichletbc(V=fs,value=T_bottom,dofs=bottom_dofs)

        return [bc_bottom,bc_top]




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