from dolfinx.fem import Function, FunctionSpace,locate_dofs_geometrical
import numpy as np

def set_IC_T(fs: FunctionSpace, T: Function) -> None:
    fs_T, map_T = fs.sub(0).collapse()
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

def set_IC_u(fs: FunctionSpace, u: Function) -> None:
    fs_u, map_u = fs.sub(5).collapse()
    shield_gas_height = 0.3
    u_dofs = locate_dofs_geometrical(fs_u, lambda x: x[2] >= shield_gas_height)
    # Array indexing with another array only works in numpy
    arr_map_u = np.array(map_u)
    u.x.array[arr_map_u[u_dofs]] = 0.
    u.x.array[arr_map_u[u_dofs]][1] = 1.
    return

def set_IC_phases(fs: FunctionSpace, gas: Function, solid: Function) -> None:
    fs_g, map_g = fs.sub(4).collapse()
    fs_s, map_s = fs.sub(2).collapse()
    # Array indexing with another array only works in numpy
    arr_map_g = np.array(map_g)
    arr_map_s = np.array(map_s)

    # except for the solid regions, anywhere else there is shielding gas
    gas.x.array[arr_map_g] = 1.

    solid_height = 0.2
    solid_dofs = locate_dofs_geometrical(fs_s, lambda x: x[2] <= solid_height)
    gas_dofs = locate_dofs_geometrical(fs_g, lambda x: x[2] <= solid_height)
    solid.x.array[arr_map_s[solid_dofs]] = 1.
    gas.x.array[arr_map_g[gas_dofs]] = 0.
    return

def project_initial_conditions(fs: FunctionSpace, f: Function) -> None:
    T, _, a_s, _, a_g, u = f.split()   

    # Temperature
    set_IC_T(fs,T)

    # Velocity
    set_IC_u(fs,u)

    # Phase fractions
    set_IC_phases(fs, a_s, a_g)
    
    return