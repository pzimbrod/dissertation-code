from firedrake import (Function, conditional, SpatialCoordinate,
                       as_vector, Constant)
import numpy as np

class ICs:
    def _set_IC_T(self, T: Function) -> None:
        mesh = self.mesh
        x, y, z = SpatialCoordinate(mesh)
        ## Ambient temperature
        T_ambient = 273.
        ## Pre-heated, solidified material
        T_base = 498
        solid_height = 0.2
        expr = conditional( z <= solid_height, T_base, T_ambient)
        T.interpolate(expr)
        return

    def _set_IC_u(self, u: Function) -> None:
        mesh = self.mesh
        x, y, z = SpatialCoordinate(mesh)
        shield_gas_height = 0.3

        expr = conditional( z <= shield_gas_height, 
                           as_vector([0.0,0.0,0.0]),
                           as_vector([0.0,1.0,0.0])
        )
        u.interpolate(expr)
        return

    def _set_IC_phases(self, gas: Function, solid: Function) -> None:
        mesh = self.mesh
        x, y, z = SpatialCoordinate(mesh)

        solid_height = 0.2
        expr_solid = conditional( z <= solid_height, 1.0, 0.0)
        expr_gas = conditional( z <= solid_height, 0.0, 1.0)
        solid.interpolate(expr_solid)
        gas.interpolate(expr_gas)
        return