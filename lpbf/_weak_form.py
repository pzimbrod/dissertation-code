from ufl import (grad, div, jump, avg, inner, dot, dx, ds, dS, FacetNormal)
from dolfinx.fem import (Constant)
#from firedrake import (FacetNormal, split, Constant)
from _fe_operators import FEOperator

class WeakForm(FEOperator):
    def _assemble_weak_form(self) -> None:
        # Building the weak form using UFL must be done with split
        # https://www.firedrakeproject.org/demos/camassaholm.py.html
        self.residual_form = 0
        self.n = FacetNormal(self.mesh)
        self.__assemble_thermal_problem()
        self.__assemble_phase_problem()
        self.__assemble_pressure_problem()
        self.__assemble_velocity_problem()

    def __assemble_thermal_problem(self) -> None:
        kappa = 1
        T_p = self.solution.previous.sub[5]
        T_n = self.solution.next.sub[5]
        test = self.testFunctions[5]
        eltype = self.config["T"]["element"]
        f = Constant(0.0)
        self.residual_form += (
            # Mass Matrix
            self._time_derivative(test=test,u_previous=T_p,u_next=T_n)
            # Laplacian
            + kappa * self._laplacian(type=eltype,test=test,u=T_n)
            # Right hand side
            - f * test * dx
        )
    
    def __assemble_phase_problem(self) -> None:
        solid_p, liquid_p, gas_p, _, u_p, _ = self.solution.previous.split()
        solid_n, liquid_n, gas_n, _, u_n, _ = self.solution.next.split()
        test_s, test_l, test_g, _, test_u, _ = self.testFunctions
        u_solid, u_liquid, u_gas = u_n * solid_n, u_n * liquid_n, u_n * gas_n
        eltype = self.config["alphas"]["element"]

        # Solid fraction
        self.residual_form += (
            # Mass Matrix
            self._time_derivative(test=test_s, u_previous=solid_p,u_next=solid_n)
            # Advection
            - self._divergence(type=eltype,test=test_s,u=u_solid,numerical_flux=self._upwind_vector)
        )
        # Liquid fraction
        self.residual_form += (
            # Mass Matrix
            self._time_derivative(test=test_l, u_previous=liquid_p,u_next=liquid_n)
            # Advection
            - self._divergence(type=eltype,test=test_l,u=u_liquid,numerical_flux=self._upwind_vector)
        )
        # Gas fraction
        self.residual_form += (
            # Mass Matrix
            self._time_derivative(test=test_g, u_previous=gas_p,u_next=gas_n)
            # Advection
            - self._divergence(type=eltype,test=test_g,u=u_gas,numerical_flux=self._upwind_vector)
        )

    def __assemble_pressure_problem(self) -> None:
        p = self.solution.next.sub(3)
        # We need a vector valued test function
        test_u = self.testFunctions[4]
        # Pressure constraint does not have any sense of wind, thus
        # no flux term
        # c.f. https://arxiv.org/pdf/2203.14881.pdf
        eltype = "CG"

        self.residual_form += self._gradient(type=eltype,test=test_u,u=p,numerical_flux=self._upwind_scalar)
    
    def __assemble_velocity_problem(self) -> None:
        p_n, u_n = self.solution.next.sub(3), self.solution.next.sub(4)
        u_p = self.solution.previous.sub(4)
        test_p, test_u = self.testFunctions[3], self.testFunctions[4]
        eltype = self.config["u"]["element"]
        self.residual_form += (
            # mass matrix
            self._time_derivative(test=test_u, u_previous=u_p, u_next=u_n)
            # continuity
            + self._divergence(type=eltype, test=test_p, u=u_n)
            # viscosity
            - self._laplacian(type=eltype, test=test_u, u=u_n)
        )
