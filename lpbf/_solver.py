from firedrake import (NonlinearVariationalProblem,
                    NonlinearVariationalSolver, solve, VectorSpaceBasis,
                    MixedVectorSpaceBasis)

class Solver:

    def _assemble_problem(self) -> None:
        print("Assembling problem")
        self.prob = NonlinearVariationalProblem(F=self.residual_form,u=self.solution.next,bcs=self.bcs)
        print("Finished assembling problem")

    def _assemble_solver(self, solver_parameters: dict | None) -> None:
        if solver_parameters is None:
            solver_parameters={'snes_type': 'newtonls',
                            'ksp_type': 'preonly',
                            'pc_type': 'lu'}
        print("Assembling solver")
        fs = self.function_space
        v_basis = VectorSpaceBasis(constant=True)
        nullspace = MixedVectorSpaceBasis(fs,[v_basis,
                                              v_basis,
                                              v_basis,
                                              fs.sub(3),
                                              v_basis,
                                              v_basis])
        self.solver = NonlinearVariationalSolver(self.prob)
        print("Finished assembling solver")

    def solve(self):
        self.solver.solve()