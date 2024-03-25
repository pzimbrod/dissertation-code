from dolfinx.fem import Function
import numpy as np
from .FEData import FEData

class InitialConditions:
    """
    A collection of functions that specify the initial configuration
    of all fields.

    Attributes
    ----------
    `functions` : `dict[str,function]`
        for each field, contains the function that interpolates
        the respective intital condition


    Methods
    -------
    apply(fe_data: FEData)
        for each function present in `fe_data`, interpolate the
        respective IC
    """

    def __init__(self) -> None:
        self.functions = {
            "alpha_solid":  self._alpha_solid,
            "alpha_liquid": self._alpha_liquid,
            "alpha_gas":    self._alpha_gas,
            "p":            self._p,
            "u":            self._u,
            "T":            self._T
        }
        
        return
    

    def apply(self, fe_data: FEData) -> None:
        """
        for each function present in `fe_data`, interpolate the
        respective IC
        
        Parameters
        ----------

        `fe_data` : `FEData`
            the data structure containing Finite Elemenent specific
            attributes of the problem
        """
        for (field,ic_fun) in self.functions.items():
            if field in fe_data.config.keys():
                ic_fun(fe_data=fe_data)
        
        fe_data.solution.update()
        
        return


    def _alpha_solid(self, fe_data: FEData) -> None:
        solid = fe_data.solution["alpha_solid"].current
        def IC_solid(x,height=0.2):
            return np.where(x[2] <= height, 1.0, 0.0)

        solid.interpolate(IC_solid)
        return
    

    def _alpha_liquid(self, fe_data: FEData) -> None:
        liquid = fe_data.solution["alpha_solid"].current
       
        def IC_liquid(x):
            return np.zeros((1,x.shape[1]))

        liquid.interpolate(IC_liquid)
        return
    

    def _alpha_gas(self, fe_data: FEData) -> None:
        gas = fe_data.solution["alpha_solid"].current

        def IC_gas(x,height=0.2):
            return np.where(x[2] > height, 1.0, 0.0)

        gas.interpolate(IC_gas)
        return


    def _u(self, fe_data: FEData) -> None:
        u = fe_data.solution["u"].current
        dim = u.ufl_shape[0]

        def IC_u(x, height=0.3):
            values = np.zeros((dim,x.shape[1]))
            cond = np.argwhere(x[2] > height)
            values[1,cond] = 1.0
            return values
        
        u.interpolate(IC_u)
        return


    def _p(self, fe_data: FEData) -> None:
        p = fe_data.solution["p"].current

        def IC_p(x):
            return np.zeros((1,x.shape[1]))
        
        p.interpolate(IC_p)
        return



    def _T(self, fe_data: FEData) -> None:
        T = fe_data.solution["T"].current

        def IC_T(x, T_ambient=298.0, T_base = 498.0, height = 0.2):
            return np.where(x[2] <= height, T_base, T_ambient)

        T.interpolate(IC_T)
        return
