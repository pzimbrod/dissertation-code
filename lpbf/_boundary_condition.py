from firedrake import (DirichletBC,Constant)

class BCs:
    """
    Implements the following boundary conditions:

                  | inlet           outlet          bottom          walls
    ----------------------------------------------------------------------------
    alpha_solid   | 0 Dirichlet     0 Neumann       1 Dirichlet     0 Neumann
    alpha_liquid  | 0 Dirichlet     0 Neumann       0 Dirichlet     0 Neumann
    alpha_gas     | 1 Dirichlet     0 Neumann       0 Dirichlet     0 Neumann
    p             | 0 Neumann       0 Dirichlet     0 Neumann       0 Neumann
    u             | 1 Dirichlet     0 Neumann       0 Neumann       0 Neumann
    T             | 0 Neumann       0 Neumann       473 Dirichlet   0 Neumann
    """

    def _setup_bcs(self):
        bc_solid = self.__setup_solid_bc()
        bc_liquid = self.__setup_liquid_bc()
        bc_gas = self.__setup_gas_bc()
        bc_p = self.__setup_pressure_bc()
        bc_u = self.__setup_velocity_bc()
        bc_T = self.__setup_temperature_bc()
        return [*bc_solid,*bc_liquid,*bc_gas,*bc_p,*bc_u,*bc_T]
    
    def __setup_solid_bc(self):
        fs = self.function_space.sub(0)
        val_inlet = Constant(0.0)
        bc_inlet = DirichletBC(V=fs,g=val_inlet,
                         sub_domain=self.bc_markers["inlet"])
        val_bottom = Constant(1.0)
        bc_bottom = DirichletBC(V=fs,g=val_bottom,
                         sub_domain=self.bc_markers["bottom"])
        return [bc_inlet,bc_bottom]
    
    def __setup_liquid_bc(self):
        fs = self.function_space.sub(1)
        val_inlet = Constant(0.0)
        bc_inlet = DirichletBC(V=fs,g=val_inlet,
                         sub_domain=self.bc_markers["inlet"])
        val_bottom = Constant(0.0)
        bc_bottom = DirichletBC(V=fs,g=val_bottom,
                         sub_domain=self.bc_markers["bottom"])
        return [bc_inlet,bc_bottom]

    def __setup_gas_bc(self):
        fs = self.function_space.sub(2)
        val_inlet = Constant(1.0)
        bc_inlet = DirichletBC(V=fs,g=val_inlet,
                         sub_domain=self.bc_markers["inlet"])
        val_bottom = Constant(0.0)
        bc_bottom = DirichletBC(V=fs,g=val_bottom,
                         sub_domain=self.bc_markers["bottom"])
        return [bc_inlet,bc_bottom]

    def __setup_pressure_bc(self):
        fs = self.function_space.sub(3)
        val = Constant(0.0)
        bc = DirichletBC(V=fs,g=val,
                         sub_domain=self.bc_markers["outlet"])
        return [bc]
    
    def __setup_velocity_bc(self):
        fs = self.function_space.sub(4)
        val = Constant((0.0,1.0, 0.0))
        bc = DirichletBC(V=fs,g=val,
                         sub_domain=self.bc_markers["inlet"])
        return [bc]
    
    def __setup_temperature_bc(self):
        fs = self.function_space.sub(5)
        val = Constant(473.0)
        bc = DirichletBC(V=fs,g=val,
                         sub_domain=self.bc_markers["bottom"])
        return [bc]
