from ufl import (grad, div, jump, avg, inner, dot, dx, ds, dS)
from firedrake import (FacetNormal)
from _dg import DGMethod

class WeakForm(DGMethod):
    def _setup_weak_form(self) -> None:
        self.residual_form = 0
        self.n = FacetNormal(self.mesh)
        self.__setup_thermal_problem()
        self.__setup_phase_problem()

    def __setup_thermal_problem(self) -> None:
        kappa = 1
        T = self.functions[5]
        test = self.testFunctions[5]
        f = 1
        self.residual_form += (
            # Mass Matrix
            (T.next - T.previous) * test * dx
            + self.dt * (
            # Laplacian
            - kappa * dot(grad(T.next),grad(test)) * dx
            # Right hand side
            - f * test * dx
            )
        )
    
    def __setup_phase_problem(self) -> None:
        solid, liquid, gas, _, u, _ = self.functions
        test_s, test_l, test_g, _, test_u, _ = self.testFunctions

        # Solid fraction
        self.residual_form += (
            # Mass Matrix
            (solid.next - solid.previous) * test_s * dx
            + self.dt * (
            # Advection
            - self._DG_div(test_s,velocity=u.next, u=solid.next,numerical_flux=self._upwind)
            )
        )
        # Liquid fraction
        self.residual_form += (
            # Mass Matrix
            (liquid.next - liquid.previous) * test_l * dx
            + self.dt * (
            # Advection
            - self._DG_div(test_l,velocity=u.next, u=liquid.next,numerical_flux=self._upwind)
            )
        )
        # Gas fraction
        self.residual_form += (
            # Mass Matrix
            (gas.next - gas.previous) * test_g * dx
            + self.dt * (
            # Advection
            - self._DG_div(test_g,velocity=u.next, u=gas.next,numerical_flux=self._upwind)
            )
        )