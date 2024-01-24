from firedrake import (NonlinearVariationalProblem, NonlinearVariationalSolver, solve)

class Solver:

    def _assemble_problem(self):
        self.prob = NonlinearVariationalProblem(F=self.residual_form,u=self.solution.next,bcs=self.bcs)

    def _assemble_solver(self):
        solver_parameters={"mat_type": "aij",
                            "snes_monitor": None,
                            "ksp_type": "gmres",
                            "pc_type": "lu",
                            "pc_factor_mat_solver_type": "mumps"}
        self.solver = NonlinearVariationalSolver(self.prob,solver_parameters=solver_parameters)

    def solve(self):
        self.solver.solve()