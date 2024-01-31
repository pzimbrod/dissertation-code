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
        if solver_parameters is None:
            solver_parameters={'snes_type': 'newtonls',
                            'ksp_type': 'preonly',
                            'pc_type': 'lu'}
        self.solver = nls.petsc.NewtonSolver(self.mesh.comm,self.prob)

    def solve(self):
        n, converged = self.solver.solve(self.solution.next)
        assert(converged)