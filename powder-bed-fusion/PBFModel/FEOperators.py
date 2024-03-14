from ufl import (dot, inner, grad, jump, avg, div, dx, ds, dS, 
                 Form, TestFunction,FacetNormal)
from dolfinx.fem import (Function)
from .TimeDependentFunction import TimeDependentFunction

class FEOperators:
    """
    Implements all methods necessary to formulate Continuous (CG) and
    Discontinuous Galerkin (DG) weak formulations.
    """

    def time_derivative(self, test: TestFunction, u_previous: Function,
                        u_current: Function, dt: float,
                        coefficient=None) -> Form:
        """
        Computes the mass matrix of a function `u` using the `TimeDependentFunction`
        class, which holds the values of `u` at different time steps.
        """
        if coefficient == None:
            F = inner(test, u_current - u_previous) / dt * dx
        else:
            F = coefficient * inner(test, u_current - u_previous) / dt * dx

        return F

    
    def divergence(self, type: str, test: TestFunction, u: Function,
                numerical_flux=None) -> Form:
        """
        Computes the divergence of an expression `(velocity * u)` in DG form where `velocity` is a vector and `u` is a scalar.
        """
        F = - inner(grad(test), u) * dx # Partial integration
        if type == "Lagrange":
            pass    # Nothing else to do
        elif type == "Discontinuous Lagrange":
            n = self.n
            F += jump(test) * numerical_flux(u,n) * dS # Hull integral
        else:
            raise NotImplementedError("Unknown type of discretization")

        return F

    
    def gradient(self, type: str, test: TestFunction, u: Function, numerical_flux=None) -> Form:
        """
        Computes the gradient of a field `u`.
        """
        F = - inner(div(test),u) * dx
        if type == "Lagrange":
            pass    # Nothing else to do
        elif type == "Discontinuous Lagrange":
            n = self.n
            F += (
                dot(jump(test),numerical_flux(n, u)) * dS
            )
        else:
            raise NotImplementedError("Unknown type of discretization")

        return F


    def laplacian(self, type: str, test: TestFunction, u: Function,
                   coefficient=None, numerical_flux=None) -> Form:
        """
        Computes the laplacian of a field `u`.
        As the order of this operator is two, there is no DG implementation.
        """
        F = 0
        if type == "Discontinuous Lagrange":
            raise TypeError("Laplacian is not defined for DG discretizations. The operator needs to be hybridized first.")
        elif type == "Lagrange":
            if coefficient == None:
                F -= inner(grad(test),grad(u)) * dx
            else:
                F -= inner(grad(test),coefficient*grad(u)) * dx
        else:
            raise NotImplementedError("Unknown type of discretization")

        return F


    def upwind_vector(self,n: FacetNormal, u: Function) -> Form:
        """
        Returns the DG upwind flux of an expression `(velocity * u)` 
        that is compatible with unstructured meshes, i.e. it is expressed
        in terms of jumps and averages.
        """
        u_n = 0.5*(dot(u, n) + abs(dot(u, n)))

        return jump(u_n)
    

    def upwind_scalar(self,n: FacetNormal, u: Function) -> Form:
        """
        Returns the DG upwind flux of an expression `u` 
        that is compatible with unstructured meshes, i.e. it is expressed
        in terms of jumps and averages.
        """
        flux = 0.5*(u * n + abs(u * n))

        return jump(flux)


    def lax_friedrichs(self,velocity: Function,n: FacetNormal, 
                       u: Function) -> Form:
        v_max = max(max(velocity),0)

        return dot(avg(velocity * u),n('+')) + 0.5 * v_max * jump(u)


    def HLLE(self,velocity: Function,n: FacetNormal, u: Function) -> Form:
        v_max = max(max(velocity),0)
        v_min = min(min(velocity),0)

        return v_max/(v_max-v_min) * dot(velocity('+'),n('+'))*u('+') - \
            v_min * dot(velocity('-'),n('+'))*u('-') - v_max * v_min * jump(u)
