from ufl import (grad, div, inner, dot, dx, ds, dS)

class WeakForm:
    def _setup_weak_form(self):
        self.residual_form = 0
        self.__setup_thermal_problem()

    def __setup_thermal_problem(self):
        kappa = 1
        T_next = self.functions[5].next
        T_previous = self.functions[5].previous
        test = self.testFunctions[5]
        f = 1
        self.residual_form += (
            # Mass Matrix
            (T_next - T_previous) * test * dx
            + self.dt * (
            # Laplacian
            - kappa * dot(grad(T_next),grad(test)) * dx
            # Right hand side
            - f * test * dx
            )
        )