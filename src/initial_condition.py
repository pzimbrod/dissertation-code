from dolfinx.fem import Function, FunctionSpace,locate_dofs_geometrical
import numpy as np

def set_IC_T(fs_T: FunctionSpace, map_T,T: Function) -> None:
    ## Ambient temperature
    T_ambient = 273.
    T.x.array[:] = T_ambient
    ## Pre-heated, solidified material
    T_base = 498
    solid_height = 0.2
    solid_dofs = locate_dofs_geometrical(fs_T, lambda x: x[2] <= solid_height)
    # Array indexing with another array only works in numpy
    arr_map_T = np.array(map_T)
    T.x.array[arr_map_T[solid_dofs]] = T_base
    return

def set_IC_u(fs_u: FunctionSpace, map_u, u: Function) -> None:
    block_velocity = np.vstack((0.,1.,0.))
    shield_gas_height = 0.3
    u_dofs = locate_dofs_geometrical(fs_u, lambda x: x[2] >= shield_gas_height)
    # Array indexing with another array only works in numpy
    arr_map_u = np.array(map_u)
    u.x.array[arr_map_u[u_dofs]] = 0.
    u.x.array[arr_map_u[u_dofs]][1] = 1.
    return

def project_initial_conditions(fs: FunctionSpace, f: Function) -> None:
    T, p, a_s, a_l, a_g, u = f.split()   

    # Temperature
    fs_T, map_T = fs.sub(0).collapse()
    set_IC_T(fs_T, map_T,T)

    # Velocity
    fs_u, map_u = fs.sub(5).collapse()
    set_IC_u(fs_u,map_u,u)
    
    return