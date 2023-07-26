from dolfinx.fem import Function
from ufl import Form, exp, sqrt
import numpy as np

"""
Clausius Clapeyron equation
Args:
    p0: ambient pressure
    Hv: latent heat of fusion
    R:  universal gas constant
    Tv: vaporisation temperature
    T:  temperature
"""
def recoil_pressure(constants: dict, T: Function) -> Form :
    p0, Hv, R, Tv = constants["p0"], constants["Hv"], constants["R"], constants["Tv"]
    return 0.53*p0*exp(Hv/R * (1/Tv - 1/T))

def heat_of_vaporisation(constants: dict, T: Function, recoil_pressure) -> Form:
    M, R = constants["M"], constants["R"]
    return 0.82 * recoil_pressure(constants,T)/sqrt(2*np.pi*M*R*T)

def radiation(constants: dict, T: Function) -> Form:
    T_amb, epsilon, sb = constants["T_amb"], constants["epsilon"], constants["sb"]
    return sb*epsilon*(T**4 - T_amb**4)