from dolfinx.mesh import create_interval, locate_entities_boundary
from mpi4py import MPI
from dolfinx.mesh import locate_entities_boundary
from dolfinx import fem, io, plot, nls, log
from dolfinx.io import gmshio
from dolfinx.fem import (FunctionSpace, Function, Constant, dirichletbc, 
                        locate_dofs_geometrical, form, locate_dofs_topological ,
                        assemble_scalar, VectorFunctionSpace, Expression)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, NonlinearProblem
from ufl import (TrialFunction, TestFunction, FiniteElement, grad, dot, inner, 
                lhs, rhs, Measure, SpatialCoordinate, FacetNormal)#, ds, dx
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

class ThermalProblem:
    def __init__(self,mesh_path,dt,degree=1) -> None:
        self.mesh, cell_tags, facet_tags = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, 0, gdim=1)
        self.fe = FiniteElement("P",self.mesh.ufl_cell(),degree)
        self.fs = FunctionSpace(mesh=self.mesh,element=self.fe)

        self.T_current = Function(self.fs)       # For nonlinear problems, there is no TrialFunction
        self.T_current.name = "Temperature"
        self.v = TestFunction(self.fs)
        self.T_previous = Function(self.fs)      # previous time step
        self.dt = dt

    def set_initial_condition(self, temp_value: float) -> None:
        x = SpatialCoordinate(self.mesh)
        def temp_init(x):
            values = np.full(x.shape[1], temp_value, dtype = ScalarType) 
            return values
        self.T_previous.interpolate(temp_init)
        self.T_current.interpolate(temp_init)
    
    def set_dirichlet_bc(self, bc_value: float) -> None:
        fdim = self.mesh.topology.dim - 1
        boundary_facets = locate_entities_boundary(
            self.mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
        self.bc = fem.dirichletbc(PETSc.ScalarType(bc_value), 
                                  fem.locate_dofs_topological(self.fs, fdim, boundary_facets), self.fs)

    def write_initial_output(self, output_name: str, t: float = 0.0) -> None:
        self.xdmf = io.XDMFFile(self.mesh.comm, f"{output_name}.xdmf", "w")
        self.xdmf.write_mesh(self.mesh)
        self.xdmf.write_function(self.T_current, t)
                            
    def setup_weak_form(self,parameters: dict) -> None:
        # Right hand side
        f = Constant(self.mesh,ScalarType(parameters["f"]))
        epsilon = Constant(self.mesh,ScalarType(parameters["epsilon"]))
        sigma = Constant(self.mesh,ScalarType(parameters["sigma"]))
        T_0 = Constant(self.mesh, ScalarType(parameters["T_0"]))
        alpha = Constant(self.mesh,ScalarType(parameters["alpha"]))

        ds = Measure("exterior_facet",domain=self.mesh)
        dx = Measure("dx",domain=self.mesh)
        
        self.F = (
            # Mass Matrix
            (self.T_current - self.T_previous) * self.v * dx
            + self.dt * (
            # Laplacian
            - alpha * dot(grad(self.T_current),grad(self.v)) * dx
            # Right hand side
            - f * self.v * dx
            # Radiation
            + sigma * epsilon * (self.T_current**4 - T_0**4) * self.v * ds
            # Convection
            + 0.1 * (self.T_current - T_0) * self.v * ds
            )
        )
    
    def setup_solver(self) -> None:
        self.prob = fem.petsc.NonlinearProblem(self.F,self.T_current)

        self.solver = nls.petsc.NewtonSolver(self.mesh.comm, self.prob)
        self.solver.convergence_criterion = "incremental"
        self.solver.rtol = 1e-12
        self.solver.report = True

        self.ksp = self.solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = self.ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "cg"
        opts[f"{option_prefix}pc_type"] = "gamg"
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        self.ksp.setFromOptions()
    
    def solve(self,t) -> None:
        n, converged = self.solver.solve(self.T_current)
        self.T_current.x.scatter_forward()

        # Update solution at previous time step (u_n)
        self.T_previous.x.array[:] = self.T_current.x.array[:]

        # Write solution to file
        self.xdmf.write_function(self.T_current, t)
    
    def finalize(self) -> None:
        self.xdmf.close()