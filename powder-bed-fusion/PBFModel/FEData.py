from mpi4py import MPI
from .Mesh import Mesh
from .TimeDependentFunction import TimeDependentFunction
from .FEOperators import FEOperators
from dolfinx.fem import (FunctionSpace, Function)
from ufl import (FiniteElement, VectorElement, MixedElement,
                Form, TestFunctions, TestFunction, split)

class FEData:
    """
    An abstraction holding all data structures that belong to a
    finite element specific implementation.
    """

    def __init__(self, mesh: Mesh, config: dict,
                 create_mixed: bool = False) -> None:
        self.is_mixed = create_mixed
        self.config = config
        self.finite_elements = self.__init_finite_elements(
                                            mesh=mesh)
        self.function_spaces = self.__init_function_space(
                                            mesh=mesh)
        self.testFunctions = self.__init_test_functions()
        self.solution = self.__init_solution()
        self.operators = FEOperators()

        return
    
    
    def __init_finite_elements(self, mesh: Mesh) -> (dict):
        """
        Create the mixed finite element (FE) to describe the problem.
        Each variable receives its own sub-FE according to the config
        """

        cell = mesh.dolfinx_mesh.ufl_cell()
        finite_elements = {}

        for (field, field_config) in self.config.items():
            if field_config["type"] == "scalar":
                finite_elements[field] = FiniteElement(
                    family=field_config["element"],
                    cell=cell,
                    degree=field_config["degree"]
                )
            elif field_config["type"] == "vector":
                finite_elements[field] = VectorElement(
                    family=field_config["element"],
                    cell=cell,
                    degree=field_config["degree"]
                )

        if self.is_mixed:
            out = {"mixed": MixedElement(list(finite_elements.values()))}
        else:
            out = finite_elements
        
        return out
    

    def __init_function_space(self, mesh: Mesh) -> dict:
        """
        Create a dolfinx Function Space from the mixed Finite Element
        """
        function_spaces = {}

        if self.is_mixed:
            self.mixed_function_space = FunctionSpace(
                mesh=mesh.dolfinx_mesh,
                element=self.finite_elements["mixed"]
            )
            num_spaces = self.mixed_function_space.num_sub_spaces
            for (idx,field) in zip(range(0,num_spaces),self.config.keys()):
                function_spaces[field] = self.mixed_function_space.sub(idx)
        else:
            for (field, fe) in self.finite_elements.items():
                function_spaces[field] = FunctionSpace(
                    mesh=mesh.dolfinx_mesh,
                    element=fe
                )

        return function_spaces

    
    def __init_solution(self) -> dict:
        functions = {}

        if self.is_mixed:
            num_spaces = len(self.function_spaces)
            self.sub_map = {}
            self.mixed_solution = TimeDependentFunction(previous=Function(self.mixed_function_space),
                                                        current=Function(self.mixed_function_space))
            for (idx, field) in zip(range(0,num_spaces),self.config.keys()):
                functions[field] = TimeDependentFunction(
                    previous=self.mixed_solution.previous.sub(idx),
                    current=self.mixed_solution.current.sub(idx)
                )
                self.sub_map[field] = idx
        else:
            for (field,fs) in self.function_spaces.items():
                functions[field] = TimeDependentFunction(
                    previous=Function(fs),
                    current=Function(fs)
                )
        
        return functions
    

    def __init_test_functions(self) -> dict:
        test_functions = {}

        if self.is_mixed:
            num_spaces = len(self.function_spaces)
            test_fns = TestFunctions(self.mixed_function_space)
            for (idx, field) in zip(range(0,num_spaces),self.config.keys()):
                test_functions[field] = test_fns[idx]
        else:
            for (field, fs) in self.function_spaces.items():
                test_functions[field] = TestFunction(fs)

        return test_functions
    

    def setup_weak_form(self, dt: float) -> None:
        self.weak_form = 0

        self.weak_form += self.__weak_heat_eq(dt=dt)

        return


    def __weak_heat_eq(self, dt: float) -> Form:
        if self.is_mixed:
            T_prev      = split(self.mixed_solution.previous)[self.sub_map["T"]]
            T_current   = split(self.mixed_solution.current)[self.sub_map["T"]]
        else:
            T_prev      = self.solution["T"].previous
            T_current   = self.solution["T"].current

        test = self.testFunctions["T"]
        eltype = self.function_spaces["T"].ufl_element().family()

        residual_form = (
            # Mass Matrix
            self.operators.time_derivative(test=test,
                                           u_previous=T_prev,
                                           u_current=T_current,
                                           dt=dt)
            # Laplacian
            - self.operators.laplacian(type=eltype,
                                       test=test,
                                       u=T_current)
        )

        return residual_form