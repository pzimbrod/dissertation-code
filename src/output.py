from dolfinx import io
from dolfinx.fem import Function
from dolfinx.mesh import Mesh

def write_xdmf(mesh: Mesh,functions: Function):
    xdmf = io.XDMFFile(mesh.comm, "pbf-lbm.xdmf", "w")
    xdmf.write_mesh(mesh)
    T, p, alpha_solid, alpha_liquid, alpha_gas, u = functions.split()
    T.name = "temperature"
    p.name = "pressure"
    alpha_solid.name = "solid fraction"
    alpha_liquid.name = "liquid fraction"
    alpha_gas.name = "gas fraction"
    u.name = "velocity"
    f_out = (T, p, alpha_solid, alpha_liquid,alpha_gas,u)

    for f in f_out:
        xdmf.write_function(f)
    
    return