from ufl import (dot, inner, grad, jump, avg, div, dx, ds, dS, Form)
from firedrake import (TestFunction, Function, FacetNormal)

class DGMethod:
    """
    Implements all methods necessary to formulate Discontinuous Galerkin
    (DG) weak formulations.
    """
    
    def _DG_div(self, test: TestFunction, velocity: Function, u: Function, 
                numerical_flux) -> Form:
        """
        Computes the divergence of an expression `(velocity * u)` in DG form where `velocity` is a vector and `u` is a scalar.
        """
        n = self.n
        F = (
            jump(test) * numerical_flux(velocity,n,u) * dS # Hull integral
            - inner(test, dot(velocity * u,n)) * dx        # Patrial integration
        )
        return F
    
    def _DG_grad(self, test: TestFunction, u: Function, numerical_flux) -> Form:
        """
        Computes the gradient of a field `u` in DG form.
        """
        n = self.n
        F = (
            dot(jump(test),numerical_flux(n, u)) * dS
            - inner(div(test),u) * dx
        )
        return F

    def _upwind_vector(self,velocity: Function,n: FacetNormal, u: Function) -> Form:
        """
        Returns the DG upwind flux of an expression `(velocity * u)` 
        that is compatible with unstructured meshes, i.e. it is expressed
        in terms of jumps and averages.
        """
        vel_n = 0.5*(dot(velocity, n) + abs(dot(velocity, n)))
        return jump(vel_n * u)
    
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