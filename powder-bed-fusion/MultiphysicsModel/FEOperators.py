from ufl import (dot, inner, grad, jump, avg, div, dx, ds, dS, 
                 Form, TestFunction,FacetNormal, Identity)
from dolfinx.fem import (Function)
from basix.ufl import _ElementBase
from .Mesh import Mesh
from .TimeDependentFunction import TimeDependentFunction

class FEOperators:
    """
    Implements all methods necessary to formulate Continuous (CG) and
    Discontinuous Galerkin (DG) weak formulations.
    """
    def __init__(self, mesh: Mesh) -> None:
        self.n = FacetNormal(mesh.dolfinx_mesh)
        self.I = Identity(mesh.cell_dim)

        return


    def time_derivative(self, test: TestFunction, 
                        u: TimeDependentFunction, dt: float, coefficient=None) -> Form:
        """
        Computes the mass matrix of a function `u` using the `TimeDependentFunction`
        class, which holds the values of `u` at different time steps.
        """
        if coefficient == None:
            F = inner(test, u.current - u.previous) / dt * dx
        else:
            F = coefficient * inner(test, u.current - u.previous) / dt * dx

        return F

    
    def divergence(self, fe: _ElementBase, test: TestFunction, u: Function,
                   numerical_flux=None) -> Form:
        """
        Computes the divergence of an expression `(velocity * u)` in DG form where `velocity` is a vector and `u` is a scalar.
        """
        if fe.family_name != 'P':
            raise NotImplementedError("Unknown type of discretization")
        F = - inner(grad(test), u) * dx # Partial integration, CG part
        if fe.discontinuous:
            n = self.n
            F += inner(jump(test),numerical_flux(u)) * dS # Hull integral

        return F

    
    def gradient(self, fe: _ElementBase, test: TestFunction, u: Function, 
                 numerical_flux=None) -> Form:
        """
        Computes the gradient of a field `u`.
        """
        F = - inner(div(test),u) * dx # Partial integration, CG part
        if fe.family_name != 'P':
            raise NotImplementedError("Unknown type of discretization")
        if fe.discontinuous:
            F += (
                inner(jump(test),numerical_flux(u)) * dS
            )

        return F


    def laplacian(self, fe: _ElementBase, test: TestFunction, u: Function,
                   coefficient=None, numerical_flux=None) -> Form:
        """
        Computes the laplacian of a field `u`.
        As the order of this operator is two, there is no DG implementation.
        """
        F = 0
        if fe.family_name != 'P':
            raise NotImplementedError("Unknown type of discretization")
        if fe.discontinuous:
            raise TypeError("Laplacian is not defined for DG discretizations. The operator needs to be hybridized first.")
        else:
            if coefficient == None:
                F -= inner(grad(test),grad(u)) * dx
            else:
                F -= inner(grad(test),coefficient*grad(u)) * dx

        return F


    def upwind_vector(self, u: Function) -> Form:
        """
        Returns the DG upwind flux of an expression `(velocity * u)` 
        that is compatible with unstructured meshes, i.e. it is expressed
        in terms of jumps and averages.
        """
        n = self.n
        u_n = 0.5*(dot(u, n) + abs(dot(u, n)))

        return jump(u_n)
    

    def upwind_scalar(self, u: Function) -> Form:
        """
        Returns the DG upwind flux of an expression `u` 
        that is compatible with unstructured meshes, i.e. it is expressed
        in terms of jumps and averages.
        """
        n = self.n
        flux = 0.5*(u * n + abs(u * n))

        return jump(flux)
    

    def central_scalar(self, u: Function) -> Form:
        """
        Returns the DG upwind flux of an expression `u` 
        that is compatible with unstructured meshes, i.e. it is expressed
        in terms of jumps and averages.
        """
        n = self.n

        return avg(u*n)


    def lax_friedrichs(self,velocity: Function,u: Function) -> Form:
        v_max = max(max(velocity),0)
        n = self.n

        return dot(avg(velocity * u),n('+')) + 0.5 * v_max * jump(u)


    def HLLE(self,velocity: Function, u: Function) -> Form:
        v_max = max(max(velocity),0)
        v_min = min(min(velocity),0)
        n = self.n

        return v_max/(v_max-v_min) * dot(velocity('+'),n('+'))*u('+') - \
            v_min * dot(velocity('-'),n('+'))*u('-') - v_max * v_min * jump(u)
