from dolfinx.fem import Function
import numpy as np

def set_IC_phases(solid: Function, liquid: Function, gas: Function) -> None:
    def IC_solid(x,height=0.2):
        return np.where(x[2] <= height, 1.0, 0.0)
    
    def IC_liquid(x):
        return np.zeros((1,x.shape[1]))
    
    def IC_gas(x,height=0.2):
        return np.where(x[2] > height, 1.0, 0.0)

    solid.interpolate(IC_solid)
    liquid.interpolate(IC_liquid)
    gas.interpolate(IC_gas)
    return


def set_IC_u(u: Function, dim: int = 3) -> None:
    def IC_u(x, height=0.3):
        values = np.zeros((dim,x.shape[1]))
        cond = np.argwhere(x[2] > height)
        values[1,cond] = 1.0
        return values
    
    u.interpolate(IC_u)
    return


def set_IC_p(p: Function) -> None:
    def IC_p(x):
        return np.zeros((1,x.shape[1]))
    
    p.interpolate(IC_p)
    return



def set_IC_T(T: Function) -> None:
    def IC_T(x, T_ambient=298.0, T_base = 498.0, height = 0.2):
        return np.where(x[2] <= height, T_base, T_ambient)

    T.interpolate(IC_T)
    return
