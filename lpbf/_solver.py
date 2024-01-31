from dolfinx.fem.petsc import (NonlinearProblem,create_vector_nest)
from dolfinx.fem import form
from dolfinx import nls
from petsc4py import PETSc

class Solver:

    def _assemble_problem(self) -> None:
        self.prob = NonlinearProblem(F=self.residual_form,
                                     u=self.solution.next,
                                     bcs=self.bcs)

    def _assemble_solver(self, solver_parameters: dict | None) -> None:
        
        self.solver = nls.petsc.NewtonSolver(self.mesh.comm,self.prob)
        self.solver.convergence_criterion = "residual"
        self.solver.rtol = 1e-6
        self.solver.report = True
        if solver_parameters is None:
            ksp = self.solver.krylov_solver
            opts = PETSc.Options()  # type: ignore
            option_prefix = ksp.getOptionsPrefix()
            opts[f"{option_prefix}ksp_type"] = "preonly"
            opts[f"{option_prefix}pc_type"] = "lu"
            sys = PETSc.Sys()  # type: ignore
            # For factorisation prefer MUMPS, then superlu_dist, then default.
            if sys.hasExternalPackage("mumps"):
                opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
            elif sys.hasExternalPackage("superlu_dist"):
                opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
            ksp.setFromOptions()

    def solve(self):
        n, converged = self.solver.solve(self.solution.next)
        assert(converged)