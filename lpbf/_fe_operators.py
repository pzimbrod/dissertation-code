from ufl import (dot, inner, grad, jump, avg, div, dx, ds, dS, Form)
from firedrake import (TestFunction, Function, FacetNormal)
from TimeDependentFunction import TimeDependentFunction

class FEOperator:
    """
    Implements all methods necessary to formulate Continuous (CG) and
    Discontinuous Galerkin (DG) weak formulations.
    """

    def _time_derivative(self, test: TestFunction, u_previous, u_next) -> Form:
        """
        Computes the mass matrix of a function `u` using the `TimeDependentFunction`
        class, which holds the values of `u` at different time steps.
        """
        F = inner(test, u_next - u_previous) * dx
        return F
    
    def _divergence(self, type: str, test: TestFunction, u,
                numerical_flux=None) -> Form:
        """
        Computes the divergence of an expression `(velocity * u)` in DG form where `velocity` is a vector and `u` is a scalar.
        """
        n = self.n
        F = - inner(test, dot(u,n)) * dx # Partial integration
        if type == "CG":
            pass    # Nothing else to do
        elif type == "DG":
            F += jump(test) * numerical_flux(u,n) * dS # Hull integral
        else:
            raise NotImplementedError("Unknown type of discretization")
        return F
    
    def _gradient(self, type: str, test: TestFunction, u, numerical_flux=None) -> Form:
        """
        Computes the gradient of a field `u`.
        """
        F = - inner(div(test),u) * dx
        if type == "CG":
            pass    # Nothing else to do
        elif type == "DG":
            n = self.n
            F += (
                dot(jump(test),numerical_flux(n, u)) * dS
            )
        else:
            raise NotImplementedError("Unknown type of discretization")
        return F

    def _laplacian(self, type: str, test: TestFunction, u, numerical_flux=None) -> Form:
        """
        Computes the laplacian of a field `u`.
        As the order of this operator is two, there is no DG implementation.
        """
        F = 0
        if type == "DG":
            raise TypeError("Laplacian is not defined for DG discretizations. The operator needs to be hybridized first.")
        elif type == "CG":
            F -= inner(grad(test),grad(u)) * dx
        else:
            raise NotImplementedError("Unknown type of discretization")
        return F


    def _upwind_vector(self,n: FacetNormal, u: Function) -> Form:
        """
        Returns the DG upwind flux of an expression `(velocity * u)` 
        that is compatible with unstructured meshes, i.e. it is expressed
        in terms of jumps and averages.
        """
        u_n = 0.5*(dot(u, n) + abs(dot(u, n)))
        return jump(u_n)
    
    def _upwind_scalar(self,n: FacetNormal, u: Function) -> Form:
        """
        Returns the DG upwind flux of an expression `u` 
        that is compatible with unstructured meshes, i.e. it is expressed
        in terms of jumps and averages.
        """
        flux = 0.5*(u * n + abs(u * n))
        return jump(flux)

    def _lax_friedrichs(self,velocity: Function,n: FacetNormal, u: Function) -> Form:
        v_max = max(max(velocity),0)
        return dot(avg(velocity * u),n('+')) + 0.5 * v_max * jump(u)

    def _HLLE(self,velocity: Function,n: FacetNormal, u: Function) -> Form:
        v_max = max(max(velocity),0)
        v_min = min(min(velocity),0)
        return v_max/(v_max-v_min) * dot(velocity('+'),n('+'))*u('+') - \
            v_min * dot(velocity('-'),n('+'))*u('-') - v_max * v_min * jump(u)
