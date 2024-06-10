from mpi4py import MPI
from .Mesh import Mesh
from .TimeDependentFunction import TimeDependentFunction
from .FEOperators import FEOperators
from .MaterialModel import AbstractMaterialModel
from dolfinx.fem import (FunctionSpace, Function, Expression, functionspace)
from ufl import (Form, TestFunctions, TestFunction, split,
                inner, dx)
from basix.ufl import (element, mixed_element, _BasixElement)
from typing import Any

class AbstractFEData:
    """
    An abstraction holding all data structures that belong to a
    finite element specific implementation.
    ...

    Attributes
    ----------
    `is_mixed` : `bool`
        whether the Finite Element problem is set up using a mixed function space

    `config` : `dict[dict[str,any]]`
        the specification of how the Finite Element problem is supposed to be set up.

    `finite_elements` : `dict[str,FiniteElement]`
        the finite elements for each specified field
    
    `function_spaces` : `dict[str,FunctionSpace]`
        the function spaces for each specified field
    
    `test_functions` : `dict[str, TestFunction]`
        the test function for each specified field

    `solution` : `dict[str,TimeDependentFunction]`
        the time dependent solution variable(s) of the problem

    `operators` : `FEOperators`
        a collection of methods to express weak differential operators

    Methods
    -------
    setup_weak_form(dt: float)
        Set up the weak PDE formulation of the problem.
    """

    def __init__(self, mesh: Mesh, 
                 config: dict[str,dict[str,any]],
                 create_mixed: bool = False) -> None:
        """
        Parameters
        ----------
        `mesh` : `Mesh`
            the computational domain, including information about the boundary

        `config` : `dict[str,dict[str,any]]`
            the specification of how the Finite Element problem is supposed to be set up.
            It should have the following structure: the top level keys are the fields to be created.
            Then, for each field, the keys `element` (element type - `"CG" or "DG"`), `degree` (Finite
            Element degree - `int`) and `type` (tensor dimension - `"scalar"` or `"vector"`) must be given.
            The key "time_scheme" is optional and specifies the time stepping scheme to be used for that
            quantity. Choose between `"implicit euler"` and `"explicit euler"`.
        
        `create_mixed` : `bool`, optional
            Whether to create the problem in a mixed space. Setting this to `True` will create a monolithic
            problem that is solved at once using a Newton method, which can take considerably longer than
            a split problem, but doesn't require any specific type of solver setup that is tailored to
            the PDE problem.
        """
        self.is_mixed = create_mixed
        self.mesh = mesh.dolfinx_mesh
        self.config = config
        self.finite_elements = self.__init_finite_elements(
                                            mesh=mesh)
        self.function_spaces = self.__init_function_space(
                                            mesh=mesh)
        self.test_functions = self.__init_test_functions()
        self.solution = self.__init_solution()
        self.operators = FEOperators(mesh=mesh)

        return
    
    
    def __init_finite_elements(self, mesh: Mesh) -> dict[str,_BasixElement]:
        """
        Create the mixed finite element (FE) to describe the problem.
        Each variable receives its own sub-FE according to the config
        """

        #cell = mesh.dolfinx_mesh.ufl_cell()
        cell = mesh.dolfinx_mesh.basix_cell()
        dim = mesh.cell_dim
        finite_elements = {}

        for (field, field_config) in self.config.items():
            if field_config["type"] == "scalar":
                finite_elements[field] = element(
                    family=field_config["element"],
                    cell=cell,
                    degree=field_config["degree"]
                )
            elif field_config["type"] == "vector":
                finite_elements[field] = element(
                    family=field_config["element"],
                    cell=cell,
                    degree=field_config["degree"],
                    shape=(dim,)
                )

        if self.is_mixed:
            finite_elements["mixed"] = mixed_element(list(finite_elements.values()))
        
        return finite_elements
    

    def __init_function_space(self, mesh: Mesh) -> dict[str,FunctionSpace]:
        """
        Create a dolfinx Function Space from the mixed Finite Element
        """
        function_spaces = {}

        if self.is_mixed:
            self.mixed_function_space = functionspace(
                mesh=mesh.dolfinx_mesh,
                element=self.finite_elements["mixed"]
            )
            num_spaces = self.mixed_function_space.num_sub_spaces
            for (idx,field) in zip(range(0,num_spaces),self.config.keys()):
                function_spaces[field] = self.mixed_function_space.sub(idx)
        else:
            for (field, fe) in self.finite_elements.items():
                function_spaces[field] = functionspace(
                    mesh=mesh.dolfinx_mesh,
                    element=fe
                )

        return function_spaces
    

    
    def __init_solution(self) -> dict[str,TimeDependentFunction]:
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
    

    def __init_test_functions(self) -> dict[str,TestFunction]:
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


    def count_dofs(self) -> str:
        total_dofs = 0
        out_str = ""
        for (key,fun) in self.solution.items():
            if self.is_mixed:
                dof_count = fun.previous.collapse().x.array.shape[0]
            else:
                dof_count = fun.previous.x.array.shape[0]
            out_str += f"{key}:    {dof_count:,}\n"
            total_dofs += dof_count
        
        out_str += f"Total:    {total_dofs:,}"
        
        return out_str


    def get_time_scheme(self, key:str) -> str:
        """
        Retrieve the scheme to be used for temporal discretization for a given field
        
        Parameters
        ----------
        
        `key` : `str`
            The dict key for the respective field
        """
        if "time_scheme" in self.config[key].keys():
            return self.config[key]["time_scheme"]
        else:
            return self.default_time_schemes[key]

    
    def get_functions(self, key:str) -> tuple[Function]:
        """
        Retrieve the current and previous functions of a given field
        
        Parameters
        ----------
        
        `key` : `str`
            The dict key for the respective field
        """
        if self.is_mixed:
            # One cannot take subfunctions here, as a ufl expression is required.
            # This can only be attained using the split() method
            prev      = split(self.mixed_solution.previous)[self.sub_map[key]]
            current   = split(self.mixed_solution.current)[self.sub_map[key]]
        else:
            prev      = self.solution[key].previous
            current   = self.solution[key].current
 
        return prev, current
    

    def get_function(self, key: str, 
                        temporal_scheme: str) -> Function:
        """
        Given a field and temporal scheme, return the correct part of the corresponding
        `TimedependentFunction`
        
        Parameters
        ----------
        
        `key` : `str`
            The dict key for the respective field
        `temporal_scheme` : `str`
        """
        
        prev, current = self.get_functions(key=key)
        
        if temporal_scheme == "implicit euler":
            fn = current
        elif temporal_scheme == "explicit euler":
            fn = prev
        else:
            raise NotImplementedError(f"Time stepping scheme \
                '{temporal_scheme}' not implemented. Choose between \
                'implicit euler' and 'explicit euler'")
    
        return fn
    

    def _weak_heat_eq(self, dt: float, time_scheme: str,
                       material_model: AbstractMaterialModel) -> Form:
        
        T = self.get_function(key="T", temporal_scheme=time_scheme)
        T_prev, T_current = self.get_functions(key="T")

        rho = material_model.expressions["rho"]
        cp = material_model.expressions["cp"]
        kappa = material_model.expressions["kappa"]

        test = self.test_functions["T"]
        fe = self.finite_elements["T"]

        residual_form = (
            # Mass Matrix
            self.operators.time_derivative(test=test,
                                           u_previous=T_prev,
                                           u_current=T_current,
                                           dt=dt,
                                           coefficient=rho*cp)
            # Laplacian
            - self.operators.laplacian(fe=fe,
                                       test=test,
                                       u=T,
                                       coefficient=kappa)
        )

        return residual_form

    def _weak_advection_eq(self, dt: float, phase_key: str, time_scheme: str) -> Form:
        alpha_prev, alpha_current   = self.get_functions(key=phase_key)
        alpha = self.get_function(key=phase_key,temporal_scheme=time_scheme)
        u     = self.get_function(key="u",temporal_scheme=time_scheme)

        test = self.test_functions[phase_key]
        fe = self.finite_elements[phase_key]

        residual_form = (
            # Mass Matrix
            self.operators.time_derivative(test=test,
                                           u_previous=alpha_prev,
                                           u_current=alpha_current,
                                           dt=dt)
            # Advection
            + self.operators.divergence(fe=fe,
                                        test=test,
                                        u= u * alpha,
                                        numerical_flux=self.operators.upwind_vector)
        )

        return residual_form


    def _weak_pressure_eq(self, material_model: AbstractMaterialModel) -> Form:
        p = self.get_function("p", temporal_scheme="explicit euler")

        # for the weak gradient, a vector test function is required
        test_u = self.test_functions["u"]
        fe = self.finite_elements["p"]
        flux = self.operators.central_scalar


        residual_form = self.operators.gradient(fe=fe, test=test_u, u=p,
                                                numerical_flux=flux)        
        
        return residual_form
    

    def _weak_stokes_eq(self, dt: float, material_model: AbstractMaterialModel) -> Form:
        field_key = "u"
        u_prev, u_current = self.get_functions(key=field_key)

        test = self.test_functions[field_key]
        rho = material_model.expressions["rho"]
        fe = self.finite_elements["u"]

        residual_form = (
            # Mass Matrix
            self.operators.time_derivative(test=test,u_previous=u_prev,
                                           u_current=u_current, dt=dt,
                                           coefficient=rho)
            # Laplacian
            - self.operators.laplacian(fe=fe,
                                       test=test,
                                       u=u_prev,
                                       coefficient=None)
        )

        return residual_form
    

    def _weak_capillary_pressure(self, material_model: AbstractMaterialModel) -> Form:
        T       = self.get_function("T", temporal_scheme=self.get_time_scheme(key="T"))
        alpha1  = self.get_function("alpha1", temporal_scheme=self.get_time_scheme(key="alpha1"))
        alpha2  = self.get_function("alpha2", temporal_scheme=self.get_time_scheme(key="alpha2"))

        capillary_pressure = material_model.capillary_stress_tensor(
            I=self.operators.I,
            sigma=material_model.surface_tension_coefficient(T=T),
            alpha1=alpha1,
            alpha2=alpha2
        )

        residual_form = self.operators.divergence(fe=self.finite_elements["u"],
                                                  test=self.test_functions["u"],
                                                  u=capillary_pressure)

        return residual_form




class PBFData(AbstractFEData):
    def __init__(self, mesh: Mesh, config: dict[str, dict[str, Any]], create_mixed: bool = False) -> None:
        super().__init__(mesh, config, create_mixed)

        self.default_time_schemes = {
            "alpha_solid":  "explicit euler",
            "alpha_liquid": "explicit euler",
            "alpha_gas":    "explicit euler",
            "p":            "explicit euler",
            "u":            "explicit euler",
            "T":            "implicit euler",
        }

        # Algebraic expressions that can be evaluated in a postprocessing step
        self.expressions = self.__init_expressions()


    def __init_expressions(self) -> dict[str,Expression]:
        expressions = {}

        # one of the phase fractions can be calculated by evaluating the algebraic
        # constraint $$ \sum_i \alpha_i = 1.0 $$
        expressions["alpha_gas"] = Expression(
            1.0 - self.solution["alpha_solid"].current - self.solution["alpha_liquid"].current,
            self.function_spaces["alpha_gas"].element.interpolation_points())
        
        return expressions
        

    def setup_weak_form(self, dt: float, 
                        material_model: AbstractMaterialModel) -> None:
        """
        Set up the weak PDE formulation of the problem.
        
        Parameters
        ----------
        
        `dt` : `float`
            the time step increment
        """
        self.weak_form = 0


        # Temperature
        self.weak_form += self._weak_heat_eq(dt=dt,
                            time_scheme=self.get_time_scheme("T"),
                            material_model=material_model)
        
        # Solid phase
        self.weak_form += self._weak_advection_eq(dt=dt, phase_key="alpha_solid",
                            time_scheme=self.get_time_scheme("alpha_solid"))
        
        # Liquid phase
        self.weak_form += self._weak_advection_eq(dt=dt, phase_key="alpha_liquid",
                            time_scheme=self.get_time_scheme("alpha_liquid"))
        
        # Gaseous phase is computed in postprocessing
        ##

        # Pressure
        self.weak_form += self._weak_pressure_eq(material_model=material_model)
        self.weak_form += self._weak_recoil_pressure(material_model=material_model)

        # Velocity
        self.weak_form += self._weak_stokes_eq(dt=dt,
                                                material_model=material_model)

        return  


    def _weak_recoil_pressure(self, material_model: AbstractMaterialModel) -> Form:
        T = self.get_function(key="T", temporal_scheme="explicit euler")
        test_p = self.test_functions["p"]
        residual_form = inner(test_p,material_model.recoil_pressure(T)) * dx

        return residual_form




class RBData(AbstractFEData):
    def __init__(self, mesh: Mesh, config: dict[str, dict[str, Any]], create_mixed: bool = False) -> None:
        super().__init__(mesh, config, create_mixed)

        self.default_time_schemes = {
            "alpha1":       "explicit euler",
            "alpha2":       "explicit euler",
            "p":            "explicit euler",
            "u":            "explicit euler",
            "T":            "implicit euler",
        }

        # Algebraic expressions that can be evaluated in a postprocessing step
        self.expressions = self.__init_expressions()

        return
    

    def __init_expressions(self) -> dict[str,Expression]:
        expressions = {}

        # one of the phase fractions can be calculated by evaluating the algebraic
        # constraint $$ \sum_i \alpha_i = 1.0 $$
        expressions["alpha2"] = Expression(
            1.0 - self.solution["alpha1"].current,
            self.function_spaces["alpha2"].element.interpolation_points())
        
        return expressions
    

    def setup_weak_form(self, dt: float, 
                        material_model: AbstractMaterialModel) -> None:
        """
        Set up the weak PDE formulation of the problem.
        
        Parameters
        ----------
        
        `dt` : `float`
            the time step increment
        """
        self.weak_form = 0


        # Temperature
        self.weak_form += self._weak_heat_eq(dt=dt,
                            time_scheme=self.get_time_scheme("T"),
                            material_model=material_model)
        
        # alpha1
        self.weak_form += self._weak_advection_eq(dt=dt, phase_key="alpha1",
                            time_scheme=self.get_time_scheme("alpha1"))
        
        # alpha2 is computed in postprocessing
        ##

        # Pressure
        self.weak_form += self._weak_pressure_eq(material_model=material_model)

        # Velocity
        self.weak_form += self._weak_stokes_eq(dt=dt,
                                                material_model=material_model)
        
        # Surface tension
        self.weak_form += self._weak_capillary_pressure(material_model=material_model)


        return