from ufl import (grad, div, jump, avg, inner, dot, dx, ds, dS)
from firedrake import (FacetNormal)
from _fe_operators import FEOperator

class WeakForm(FEOperator):
    def _assemble_weak_form(self) -> None:
        self.residual_form = 0
        self.n = FacetNormal(self.mesh)
        self.__assemble_thermal_problem()
        self.__assemble_phase_problem()
        self.__assemble_pressure_problem()
        self.__assemble_velocity_problem()

    def __assemble_thermal_problem(self) -> None:
        kappa = 1
        T = self.functions[5]
        test = self.testFunctions[5]
        eltype = self.types["T"]
        f = 1
        self.residual_form += (
            # Mass Matrix
            self._time_derivative(test=test,u=T)
            + self.dt * (
            # Laplacian
            + kappa * self._laplacian(type=eltype,test=test,u=T.next)
            # Right hand side
            - f * test * dx
            )
        )
    
    def __assemble_phase_problem(self) -> None:
        solid, liquid, gas, _, u, _ = self.functions
        test_s, test_l, test_g, _, test_u, _ = self.testFunctions
        u_solid, u_liquid, u_gas = u.next * solid.next, u.next * liquid.next, u.next * gas.next
        eltype = self.types["alphas"]

        # Solid fraction
        self.residual_form += (
            # Mass Matrix
            self._time_derivative(test=test_s, u=solid)
            + self.dt * (
            # Advection
            - self._divergence(type=eltype,test=test_s,u=u_solid,numerical_flux=self._upwind_vector)
            )
        )
        # Liquid fraction
        self.residual_form += (
            # Mass Matrix
            self._time_derivative(test=test_l, u=liquid)
            + self.dt * (
            # Advection
            - self._divergence(type=eltype,test=test_l,u=u_liquid,numerical_flux=self._upwind_vector)
            )
        )
        # Gas fraction
        self.residual_form += (
            # Mass Matrix
            self._time_derivative(test=test_g, u=gas)
            + self.dt * (
            # Advection
            - self._divergence(type=eltype,test=test_g,u=u_gas,numerical_flux=self._upwind_vector)
            )
        )

    def __assemble_pressure_problem(self) -> None:
        p = self.functions[3]
        # We need a vector valued test function
        test_u = self.testFunctions[4]
        eltype = self.types["p"]

        self.residual_form += self._gradient(type=eltype,test=test_u,u=p.next,numerical_flux=self._upwind_scalar)
    
    def __assemble_velocity_problem(self) -> None:
        p, u = self.functions[3], self.functions[4]
        test_p, test_u = self.testFunctions[3], self.testFunctions[4]
        eltype = self.types["u"]
        self.residual_form += (
            # mass matrix
            self._time_derivative(test=test_u, u=u)
            + self.dt * (
                # continuity
                self._divergence(type=eltype, test=test_p, u=u.next)
                # viscosity
                - self._laplacian(type=eltype, test=test_u, u=u.next)
            )
        )