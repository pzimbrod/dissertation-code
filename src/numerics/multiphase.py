from ufl.algebra import Power, Sum
from dolfinx.fem import Function
from ufl import Form, Identity, FacetNormal, grad, Dx, outer

def interface_marker(alpha: Function) -> Form:
    dim = 2
    return Power(Sum(
        *[Power(Dx(alpha,i),2) for i in range(0,dim)]),
        0.5
    )

def interface_normal(alpha: Function) -> Form:
    mag_alpha = interface_marker(alpha=alpha)
    return 1/mag_alpha * grad(alpha)

def capillary_stress(alpha: Function,n: FacetNormal) -> Form:
    (dim,) = n.ufl_shape
    I = Identity(dim=dim)
    n = interface_normal(alpha=alpha)
    return interface_marker(alpha=alpha) * (I-outer(n,n))
