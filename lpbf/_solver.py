from dolfinx.fem.petsc import (LinearProblem,NonlinearProblem,create_vector_nest)
from ufl import lhs, rhs
from dolfinx.fem import form
from dolfinx import nls
from petsc4py import PETSc

class Solver:

    def _assemble_problem(self) -> None:
        F = self.residual_form
        jit_options = {
            "cffi_extra_compile_args": ["-O3", "-march=native"]
        }
        #a, l = lhs(F), rhs(F)
        self.prob = NonlinearProblem(F=F,
                                     u=self.solution.next,
                                     bcs=self.bcs,
                                     jit_options=jit_options)

    def _assemble_solver(self) -> None:
        
        self.solver = nls.petsc.NewtonSolver(self.mesh.comm,self.prob)
        #self.solver.convergence_criterion = "residual"
        self.solver.convergence_criterion = "incremental"
        self.solver.rtol = 1e-6
        self.solver.report = True
        ksp = self.solver.krylov_solver
        opts = PETSc.Options()  # type: ignore
        option_prefix = ksp.getOptionsPrefix()
        #opts[f"{option_prefix}ksp_type"] = "cg"
        opts[f"{option_prefix}ksp_type"] = "gmres"
        opts[f"{option_prefix}pc_type"] = "gamg"
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
        self.timestep_update()
        self.time += self.dt
        self.write_output()