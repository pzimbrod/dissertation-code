from ufl import (grad, div, jump, avg, inner, dot, dx, ds, dS)
from firedrake import (FacetNormal)
from _dg import DGMethod

class WeakForm(DGMethod):
    def _assemble_weak_form(self) -> None:
        self.residual_form = 0
        self.n = FacetNormal(self.mesh)
        self.__assemble_thermal_problem()
        self.__assemble_phase_problem()
        self.__assemble_pressure_problem()

    def __assemble_thermal_problem(self) -> None:
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
    
    def __assemble_phase_problem(self) -> None:
        solid, liquid, gas, _, u, _ = self.functions
        test_s, test_l, test_g, _, test_u, _ = self.testFunctions

        # Solid fraction
        self.residual_form += (
            # Mass Matrix
            (solid.next - solid.previous) * test_s * dx
            + self.dt * (
            # Advection
            - self._DG_div(test_s,velocity=u.next, u=solid.next,numerical_flux=self._upwind_vector)
            )
        )
        # Liquid fraction
        self.residual_form += (
            # Mass Matrix
            (liquid.next - liquid.previous) * test_l * dx
            + self.dt * (
            # Advection
            - self._DG_div(test_l,velocity=u.next, u=liquid.next,numerical_flux=self._upwind_vector)
            )
        )
        # Gas fraction
        self.residual_form += (
            # Mass Matrix
            (gas.next - gas.previous) * test_g * dx
            + self.dt * (
            # Advection
            - self._DG_div(test_g,velocity=u.next, u=gas.next,numerical_flux=self._upwind_vector)
            )
        )

    def __assemble_pressure_problem(self) -> None:
        p = self.functions[3]
        # We need a vector valued test function
        test_u = self.testFunctions[4]

        self.residual_form += self._DG_grad(test=test_u,u=p.next,numerical_flux=self._upwind_scalar)