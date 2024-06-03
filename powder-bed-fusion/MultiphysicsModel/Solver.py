from dolfinx.fem.petsc import (NonlinearProblem,
                               assemble_vector,
                               apply_lifting,set_bc, assemble_matrix,
                               create_matrix, create_vector)
from ufl import derivative, TrialFunction
from dolfinx.fem import form, Function
from dolfinx.nls.petsc import NewtonSolver
from .Mesh import Mesh
from .FEData import AbstractFEData
from .BoundaryCondition import AbstractBoundaryConditions
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

class SNESProblem:
    def __init__(self, F: form, bcs,
                 solution: Function, du: TrialFunction,
                 maxIter=50):

        self._solution = solution
        self.bcs = bcs

        self.F  = F
                  
        self.J = derivative(self.F, solution, du)

        self._F = form(self.F)
        self._J = form(self.J)
        self._obj_vec = create_vector(self._F)

        # Create matrix and vector
        self.b = create_vector(self._F)
        self.A = create_matrix(self._J)

        # Solver settings
        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        self.solver.setTolerances(max_it=maxIter)
        self.solver.getKSP().setType("preonly")
        self.solver.getKSP().getPC().setType("lu")
        self.solver.getKSP().getPC().setFactorSolverType("mumps")

        self.solver.setObjective(self.obj_fun)
        self.solver.setFunction(self.F_fun, self.b)
        self.solver.setJacobian(self.J_fun, J=self.A, P=None)
        self.solver.setMonitor(lambda _, it, residual: print(it, residual))
        self.solver.atol = 1e-6
        self.solver.rtol = 1e-6

    def create_snes_solution(self) -> PETSc.Vec:
        """
        Create a petsc4py.PETSc.Vec to be passed to petsc4py.PETSc.SNES.solve.

        The returned vector will be initialized with the initial guess provided in `self._solution`.
        """
        x = self._solution.vector.copy()
        with x.localForm() as _x, self._solution.vector.localForm() as _solution:
            _x[:] = _solution
        return x

    def update_solution(self, x: PETSc.Vec) -> None:  # type: ignore[no-any-unimported]
        """Update `self._solution` with data in `x`."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with x.localForm() as _x, self._solution.vector.localForm() as _solution:
            _solution[:] = _x
            
    def obj_fun(self, snes: PETSc.SNES, x: PETSc.Vec) -> np.float64:
            """Compute the norm of the residual."""
            self.F_fun(snes, x, self._obj_vec)
            return self.b.norm()  # type: ignore[no-any-return]

    def F_fun(self, snes: PETSc.SNES, x: PETSc.Vec, F_vec: PETSc.Vec) -> None:
            """Assemble the residual."""
            self.update_solution(x)
            with F_vec.localForm() as F_vec_local:
                F_vec_local.set(0.0)
            assemble_vector(F_vec, self._F)
            apply_lifting(F_vec, [self._J], [self.bcs], x0=[x], scale=-1.0)
            F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            set_bc(F_vec, self.bcs, x, -1.0)

    def J_fun( self, snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat, P_mat: PETSc.Mat) -> None:
            """Assemble the jacobian."""
            J_mat.zeroEntries()
            assemble_matrix(J_mat, self._J, self.bcs, diagonal=1.0)
            J_mat.assemble()

    

    def solve(self, solution: Function):
        #solution_copy = self.create_snes_solution()
        self.solver.solve(None, solution.vector)
        self.update_solution(solution.vector)
        #return self._solution
        return 


class PBFSolver:

    def __init__(self, 
                 mesh: Mesh,
                 fe_data: AbstractFEData, 
                 bc_data: AbstractBoundaryConditions) -> None:
        
        if fe_data.is_mixed:
            u = fe_data.mixed_solution.current
        else:
            u = fe_data.solution["alpha_solid"].current

        self.prob = NonlinearProblem(F=fe_data.weak_form, u=u, 
                                     bcs=bc_data.bc_list)
        self.newton_solver = NewtonSolver(comm=mesh.dolfinx_mesh.comm,
                                          problem=self.prob)
        
        self.newton_solver.convergence_criterion = "incremental"
        self.newton_solver.rtol = 1e-6
        self.newton_solver.atol = 1e-6

        # We can customize the linear solver used inside the NewtonSolver by
        # modifying the PETSc options
        ksp = self.newton_solver.krylov_solver
        opts = PETSc.Options()  # type: ignore
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "cg"
        opts[f"{option_prefix}pc_type"] = "gamg"
        ksp.setFromOptions()

        return

    def postprocess(self, fe_data: AbstractFEData) -> None:
        """
        Calculates derived quantities for the current timestep,
        such as the gas volume fraction, following \sum_i \alpha_i = 1
        """
        a_gas = fe_data.solution["alpha_gas"].current
        a_gas.interpolate(fe_data.expressions["alpha_gas"])

        return
