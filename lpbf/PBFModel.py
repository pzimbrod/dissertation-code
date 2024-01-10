from firedrake import Mesh, UnitCubeMesh
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
        #self.mesh = Mesh(meshfile=mesh_path)
        self.mesh = UnitCubeMesh(10,10,10, hexahedral=False)
        self.degrees = degrees
        self.time = 0.0
    
    def setup(self,outfile,**kwargs) -> None:
        self._setup_finite_element()
        self._setup_function_space()
        self._setup_functions()
        self._create_output(filename=outfile,**kwargs)
    
    def project_initial_conditions(self) -> None:
        a_s, a_l, a_g, p, u, T = self.functions

        # Temperature
        self._set_IC_T(T=T.previous)
        # Velocity
        self._set_IC_u(u=u.previous)
        # Phase fractions
        self._set_IC_phases(gas=a_g.previous,solid=a_s.previous)
    
    