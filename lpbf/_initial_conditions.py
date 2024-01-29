from dolfinx.fem import (Function)
import numpy as np

class ICs:
    def _project_initial_conditions(self) -> None:
        # Data access must be done using `subfunctions`
        # https://www.firedrakeproject.org/demos/camassaholm.py.html
        a_s, a_l, a_g, p, u, T = self.solution.next.split()

        # Phase fractions
        self._set_IC_phases(solid=a_s,liquid=a_l,gas=a_g)
        # Velocity
        self._set_IC_u(u=u)
        # Pressure
        self._set_IC_p(p=p)
        # Temperature
        self._set_IC_T(T=T)
        self.solution.previous.x.scatter_forward()
        self.solution.next.x.scatter_forward()

    def _set_IC_phases(self, solid: Function, liquid: Function, gas: Function) -> None:
        def IC_solid(x,height=0.2):
            return np.where(x[2] <= height, 1.0, 0.0)
        
        def IC_liquid(x):
            return np.zeros((1,x.shape[1]))
        
        def IC_gas(x,height=0.2):
            return np.where(x[2] > height, 1.0, 0.0)

        solid.interpolate(IC_solid)
        liquid.interpolate(IC_liquid)
        gas.interpolate(IC_gas)

    def _set_IC_u(self, u: Function) -> None:
        def IC_u(x, height=0.3):
            dim = self.mesh.topology.dim
            values = np.zeros((dim,x.shape[1]))
            cond = np.argwhere(x[2] > height)
            values[1,cond] = 1.0
            return values
        
        u.interpolate(IC_u)

    def _set_IC_p(self, p: Function) -> None:
        def IC_p(x):
            return np.zeros((1,x.shape[1]))
        
        p.interpolate(IC_p)

    def _set_IC_T(self, T: Function) -> None:
        def IC_T(x, T_ambient=273.0, T_base = 498.0, height = 0.2):
            return np.where(x[2] <= height, T_base, T_ambient)

        T.interpolate(IC_T)
