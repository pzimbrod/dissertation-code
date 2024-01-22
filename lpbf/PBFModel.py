from firedrake import Mesh, UnitCubeMesh
from _setup import Setup
from _initial_conditions import ICs
from _output import Output
from _weak_form import WeakForm

class PBFModel(Setup,WeakForm,ICs,Output):
    """
    Top level class for the PBF-LB/M problem.
    Private (sub)-methods are defined in sub-classes that are
    inherited from.
    """
    def __init__(self,mesh_path: str,degrees: dict, types: dict, timestep: float) -> None:
        self.mesh = Mesh(meshfile=mesh_path)
        #self.mesh = UnitCubeMesh(10,10,10, hexahedral=False)
        self.degrees = degrees
        self.types = types
        self.time = 0.0
        self.dt = timestep
    
    def setup(self,outfile,**kwargs) -> None:
        self._setup_finite_element()
        self._setup_function_space()
        self._setup_functions()
        self._project_initial_conditions()
        self._create_output(filename=outfile,**kwargs)
        # Set values for t = 0 to output
        self.timestep_update()
        self.write_output()
    
    def assemble(self) -> None:
        self._assemble_weak_form()
    
    def _project_initial_conditions(self) -> None:
        a_s, a_l, a_g, p, u, T = self.functions

        # Temperature
        self._set_IC_T(T=T.next)
        # Velocity
        self._set_IC_u(u=u.next)
        # Phase fractions
        self._set_IC_phases(gas=a_g.next,solid=a_s.next)
    
    def timestep_update(self) -> None:
        for fun in self.functions:
            fun.previous.assign(fun.next)
        return