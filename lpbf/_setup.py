from ufl import (FiniteElement, VectorElement, MixedElement, TestFunctions, TestFunction)
from dolfinx.fem import (Function, FunctionSpace, Constant)
#from firedrake import (FiniteElement, VectorElement, MixedElement, MixedFunctionSpace, Function, split, TestFunctions)
from TimeDependentFunction import TimeDependentFunction
from MaterialModel import MultiphaseModel

class Setup:
    def _setup_finite_element(self) -> None:
        cell = self.mesh.ufl_cell()
        fe_alphas = FiniteElement(self.config["alphas"]["element"],cell,
                                    self.config["alphas"]["degree"])
        fe_p      = FiniteElement(self.config["p"]["element"],cell,
                                    self.config["p"]["degree"])
        fe_u      = VectorElement(self.config["u"]["element"],cell,
                                    self.config["u"]["degree"])
        fe_T      = FiniteElement(self.config["T"]["element"],cell,
                                    self.config["T"]["degree"])
        # According to https://fenicsproject.org/pub/tutorial/html/._ftut1010.html,
        # every variable needs its own FE
        """
        self.finite_element = MixedElement([
                                            fe_alphas,  # solid phase
                                            fe_alphas,  # liquid phase
                                            fe_alphas,  # gaseous phase
                                            fe_p,       # pressure
                                            fe_u,       # velocity
                                            fe_T,       # temperature
                                            ])
                                            """
        self.finite_element = fe_T

        return
        
    def _setup_function_space(self) -> None:
        self.function_space = FunctionSpace(mesh=self.mesh,
                                            element=self.finite_element)

        return

    def _setup_functions(self) -> None:
        fs = self.function_space
        #self.testFunctions = TestFunctions(fs)
        self.testFunction = TestFunction(fs)
        self.functions = self.__setup_primary_variables()

        return
    
    def __setup_primary_variables(self) -> None:
        """
        Creates the set of primary unknowns of the model.
        As the coupled problem is nonlinear, there are no TrialFunctions.
        Returns a tuple of TimeDependentFunctions.
        """
        fs = self.function_space
        f = Function(fs)
        self.solution = TimeDependentFunction(previous=f,next=f)

        return
    
    def _setup_material_model(self,model: dict) -> None:
        self.material_model = MultiphaseModel(mesh=self.mesh,model=model)
        
        return
