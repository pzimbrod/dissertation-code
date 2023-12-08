from firedrake import (FiniteElement, VectorElement, MixedElement, MixedFunctionSpace,
                       TestFunctions, Function, FunctionSpace)

class TimeDependentFunction:
     def __init__(self,fs: FunctionSpace, name: str) -> None:
        str_previous = f"{name}_previous"
        str_next = f"{name}_next"
        fn_previous = Function(fs, name=str_previous)
        fn_next = Function(fs, name=str_next)
        self.previous = fn_previous
        self.next     = fn_next

class Setup:
    def _setup_finite_element(self) -> None:
            fe_alphas = FiniteElement("DG",self.mesh.ufl_cell(),self.degrees["alphas"])
            fe_p      = FiniteElement("DG",self.mesh.ufl_cell(),self.degrees["p"])
            fe_u      = VectorElement("CG",self.mesh.ufl_cell(),self.degrees["u"])
            fe_T      = FiniteElement("CG",self.mesh.ufl_cell(),self.degrees["T"])
            # According to https://fenicsproject.org/pub/tutorial/html/._ftut1010.html,
            # every variable needs its own FE
            self.finite_element = MixedElement([
                                                fe_alphas,  # solid phase
                                                fe_alphas,  # liquid phase
                                                fe_alphas,  # gaseous phase
                                                fe_p,       # pressure
                                                fe_u,       # velocity
                                                fe_T,       # temperature
                                                ])
        
    def _setup_function_space(self) -> None:
        self.function_space = MixedFunctionSpace(spaces=self.finite_element,mesh=self.mesh)

    def _setup_functions(self) -> None:
        fs = self.function_space
        self.testFunctions = TestFunctions(fs)
        self.functions = self.__setup_primary_variables()
    
    def __setup_primary_variables(self) -> None:
         fs = self.function_space
         alpha_solid = TimeDependentFunction(fs.sub(0), name="solid_fraction")
         alpha_liquid = TimeDependentFunction(fs.sub(1), name="liquid_fraction")
         alpha_gas = TimeDependentFunction(fs.sub(2), name="gas_fraction")
         p = TimeDependentFunction(fs.sub(3), name="pressure")
         u = TimeDependentFunction(fs.sub(4), name="velocity")
         T = TimeDependentFunction(fs.sub(5), name="temperature")
         return (alpha_solid, alpha_liquid, alpha_gas, p, u, T)
