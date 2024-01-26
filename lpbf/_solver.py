from dolfinx.fem.petsc import NonlinearProblem
from dolfinx import nls
from petsc4py import PETSc

class Solver:

    def _assemble_problem(self) -> None:
        print("Assembling problem")
        self.prob = NonlinearProblem(F=self.residual_form,
                                     u=self.solution.next,
                                     bcs=self.bcs)
        print("Finished assembling problem")

    def _assemble_solver(self, solver_parameters: dict | None) -> None:
        if solver_parameters is None:
            solver_parameters={'snes_type': 'newtonls',
                            'ksp_type': 'preonly',
                            'pc_type': 'lu'}
        print("Assembling solver")
        self.solver = nls.petsc.NewtonSolver(self.mesh.comm,self.prob)
        print("Finished assembling solver")

    def solve(self):
        self.solver.solve()