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
            inner(test, dot(velocity * u,n)) * dS            # Hull integral
            - jump(test) * numerical_flux(velocity,n,u) * dS # Partial integration
        )
        return F

    def _upwind(self,velocity: Function,n: FacetNormal, u: Function) -> Form:
        """
        Returns the DG upwind flux of an expression `(velocity * u)` 
        that is compatible with unstructured meshes, i.e. it is expressed
        in terms of jumps and averages.
        """
        vel_n = 0.5*(dot(velocity, n) + abs(dot(velocity, n)))
        return jump(vel_n * u)

    def _lax_friedrichs(self,velocity: Function,n: FacetNormal, u: Function) -> Form:
        v_max = max(max(velocity),0)
        return dot(avg(velocity * u),n('+')) + 0.5 * v_max * jump(u)

    def _HLLE(self,velocity: Function,n: FacetNormal, u: Function) -> Form:
        v_max = max(max(velocity),0)
        v_min = min(min(velocity),0)
        return v_max/(v_max-v_min) * dot(velocity('+'),n('+'))*u('+') - \
            v_min * dot(velocity('-'),n('+'))*u('-') - v_max * v_min * jump(u)