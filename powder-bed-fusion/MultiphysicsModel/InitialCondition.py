from dolfinx.fem import Function
import numpy as np
from .FEData import AbstractFEData
from typing import Any


class AbstractInitialConditions:
    def __init__(self) -> None:
        self.functions = {}

        return


    def apply(self, fe_data: AbstractFEData) -> None:
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
                fe_data.solution[field].update()
        
        fe_data.mixed_solution.update()
        
        return




class PBFInitialConditions(AbstractInitialConditions):
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
    apply(fe_data:AbstractFEData)
        for each function present in `fe_data`, interpolate the
        respective IC
    """

    def __init__(self) -> None:
        super().__init__()
            
        self.functions["alpha_solid"]   = self._alpha_solid
        self.functions["alpha_liquid"]  = self._alpha_liquid
        self.functions["alpha_gas"]     = self._alpha_gas
        self.functions["p"]             = self._p
        self.functions["u"]             = self._u
        self.functions["T"]             = self._T
        
        return


    def _alpha_solid(self, fe_data: AbstractFEData) -> None:
        solid = fe_data.solution["alpha_solid"].current
        def IC_solid(x,height=0.2):
            return np.where(x[2] <= height, 1.0, 0.0)

        solid.interpolate(IC_solid)
        return
    

    def _alpha_liquid(self, fe_data: AbstractFEData) -> None:
        liquid = fe_data.solution["alpha_solid"].current
       
        def IC_liquid(x):
            return np.zeros((1,x.shape[1]))

        liquid.interpolate(IC_liquid)
        return
    

    def _alpha_gas(self, fe_data: AbstractFEData) -> None:
        gas = fe_data.solution["alpha_solid"].current

        def IC_gas(x,height=0.2):
            return np.where(x[2] > height, 1.0, 0.0)

        gas.interpolate(IC_gas)
        return


    def _u(self, fe_data: AbstractFEData) -> None:
        u = fe_data.solution["u"].current
        dim = u.ufl_shape[0]

        def IC_u(x, height=0.3):
            values = np.zeros((dim,x.shape[1]))
            cond = np.argwhere(x[2] > height)
            values[1,cond] = 1.0
            return values
        
        u.interpolate(IC_u)
        return


    def _p(self, fe_data: AbstractFEData) -> None:
        p = fe_data.solution["p"].current

        def IC_p(x):
            return np.zeros((1,x.shape[1]))
        
        p.interpolate(IC_p)
        return



    def _T(self, fe_data: AbstractFEData) -> None:
        T = fe_data.solution["T"].current

        def IC_T(x, T_ambient=298.0, T_base = 498.0, height = 0.2):
            return np.where(x[2] <= height, T_base, T_ambient)

        T.interpolate(IC_T)
        return




class RBInitialConditions(AbstractInitialConditions):
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
    apply(fe_data:AbstractFEData)
        for each function present in `fe_data`, interpolate the
        respective IC
    """

    def __init__(self, params: dict[str,Any]) -> None:
        super().__init__()
        self.parameters = params
            
        self.functions["alpha1"]  = self._alpha1
        self.functions["alpha2"]  = self._alpha2
        self.functions["p"]       = self._p
        self.functions["u"]       = self._u
        self.functions["T"]       = self._T
        
        return


    def _alpha1(self, fe_data: AbstractFEData) -> None:
        alpha1 = fe_data.solution["alpha1"].current
        a = self.parameters["a"]
        def IC_1(x):
            return np.where(np.sqrt(x[0]**2+x[1]**2) <= a, 0.0, 1.0)

        alpha1.interpolate(IC_1)
        return
    

    def _alpha2(self, fe_data: AbstractFEData) -> None:
        alpha2 = fe_data.solution["alpha2"].current
        a = self.parameters["a"]
       
        def IC_2(x):
            return np.where(np.sqrt(x[0]**2+x[1]**2) <= a, 1.0, 0.0)

        alpha2.interpolate(IC_2)
        return
    

    def _u(self, fe_data: AbstractFEData) -> None:
        u = fe_data.solution["u"].current
        dim = u.ufl_shape[0]

        def IC_u(x):
            values = np.zeros((dim,x.shape[1]))
            values[1,:] = 0.1

            return values

        
        u.interpolate(IC_u)
        return


    def _p(self, fe_data: AbstractFEData) -> None:
        p = fe_data.solution["p"].current

        def IC_p(x):
            return np.zeros((1,x.shape[1]))
        
        p.interpolate(IC_p)
        return



    def _T(self, fe_data: AbstractFEData) -> None:
        T = fe_data.solution["T"].current
        T_l = self.parameters["temp_bottom"]
        T_u = self.parameters["temp_top"]
        a = self.parameters["a"]

        def IC_T(x):
            out = np.zeros((1,x.shape[1]))
            out[:] = (T_l+T_u)/2.0 + (T_u-T_l)/(4.0*a) * x[1,:]
            return out

        T.interpolate(IC_T)
        return
