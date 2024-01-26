from dolfinx.fem import (Function)
import numpy as np

class ICs:
    def _project_initial_conditions(self) -> None:
        # Data access must be done using `subfunctions`
        # https://www.firedrakeproject.org/demos/camassaholm.py.html
        a_s, a_l, a_g, p, u, T = self.solution.next.split()

        # Temperature
        self._set_IC_T(T=T)
        # Velocity
        self._set_IC_u(u=u)
        # Phase fractions
        self._set_IC_phases(gas=a_g,solid=a_s)

    def _set_IC_T(self, T: Function) -> None:
        def IC_T(x, T_ambient=273.0, T_base = 498.0, height = 0.2):
            return np.where(x[2] <= height, T_base, T_ambient)

        T.interpolate(IC_T)
        return

    def _set_IC_u(self, u: Function) -> None:

        def IC_u(x, height=0.3):
            return np.where(x[2] <= height,
                            np.array((0.0,0.0,0.0)),
                            np.array((0.0,1.0,0.0)))

        u.interpolate(IC_u)
        return

    def _set_IC_phases(self, solid: Function, liquid: Function, gas: Function) -> None:
        def IC_solid(x,height=0.2):
            return np.where(x <= height, 1.0, 0.0)
        
        def IC_liquid(x):
            return np.array((0.0))
        
        def IC_gas(x,height=0.2):
            return np.where(x > height, 0.0, 1.0)

        solid.interpolate(IC_solid)
        liquid.interpolate(IC_liquid)
        gas.interpolate(IC_gas)
        return