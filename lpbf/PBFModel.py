from firedrake import Mesh
from _setup import Setup
from _initial_conditions import ICs
from _output import Output

class PBFModel(Setup,ICs,Output):
    """
    Top level class for the PBF-LB/M problem.
    Private (sub)-methods are defined in sub-classes that are
    inherited from.
    """
    def __init__(self,mesh_path: str,degrees: dict) -> None:
        self.mesh = Mesh(meshfile=mesh_path)
        self.degrees = degrees
        self.time = 0.0
    
    def setup(self,**kwargs) -> None:
        self._setup_finite_element()
        self._setup_function_space()
        self._setup_functions()
        self._create_output(**kwargs)
    
    def project_initial_conditions(self) -> None:
        fs = self.function_space
        a_s, a_l, a_g, u, p, T = self.functions.split()

        # Temperature
        self.set_IC_T(T)
        # Velocity
        self.set_IC_u(u)
        # Phase fractions
        self.set_IC_phases(gas=a_g,solid=a_s)
    
    