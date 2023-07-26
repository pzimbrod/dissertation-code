from ufl.algebra import Sum
from dolfinx.fem import Function
from dolfinx.mesh import Mesh
from ufl import Form, Identity, FacetNormal, grad, Dx, outer, sqrt, inner, CellVolume
from ufl.algebra import Power

"""
The coloring function delta_s mentioned in Brackbill et al. to
mark the presence of interfaces.
delta_s = |grad(alpha)|
"""
def interface_marker(grad_alpha: Form) -> Form:
    return sqrt(inner(grad_alpha,grad_alpha))

def interface_normal(alpha: Function) -> Form:
    mag_alpha = interface_marker(alpha=alpha)
    return 1/mag_alpha * grad(alpha)

def interface_gradient(alpha1: Function, alpha2: Function) -> Form:
    return alpha1 * grad(alpha2) - alpha2 * grad(alpha1)

def unit_interface_normal(alpha1: Function, alpha2: Function, constants) -> Form:
    h = constants["h"]
    grad_alpha = interface_gradient(alpha1, alpha2)
    dN = 1e-8/h
    mag_grad_alpha = sqrt(inner(grad_alpha,grad_alpha))
    return grad_alpha/(dN + mag_grad_alpha)

"""
The capillary stress tensor:
T_ij = sigma * (I - n_i n_j)
"""
def capillary_stress(alpha1: Function, alpha2: Function,T: Function ,constants) -> Form:
    sigma = constants["sigma"]
    ni = unit_interface_normal(alpha1, alpha2,constants=constants)
    I = Identity(len(ni))
    delta_s = interface_marker(grad_alpha=interface_gradient(alpha1,alpha2))
    return delta_s * sigma(T) * (I-outer(ni,ni))
