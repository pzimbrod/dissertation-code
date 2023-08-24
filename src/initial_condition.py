from dolfinx.fem import Function, FunctionSpace,locate_dofs_geometrical
import numpy as np

def set_IC_T(fs_T: FunctionSpace,T: Function) -> None:
    ## Ambient temperature
    T_ambient = 273.
    T.x.array[:] = T_ambient
    ## Pre-heated, solidified material
    T_base = 498
    solid_height = 0.2
    solid_dofs = locate_dofs_geometrical(fs_T, lambda x: x[2] <= solid_height)
    T.x.array[solid_dofs] = T_base

    return

def project_initial_conditions(fs: FunctionSpace, f: Function) -> None:
    T, p, a_s, a_l, a_g, u = f.split()   

    # Temperature
    fs_T, _ = fs.sub(0).collapse()
    set_IC_T(fs_T,T)
    

    return