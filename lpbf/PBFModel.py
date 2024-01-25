from firedrake import Mesh, UnitCubeMesh
from _setup import Setup
from _initial_conditions import ICs
from _output import Output
from _weak_form import WeakForm
from _boundary_condition import BCs
from _solver import Solver

class PBFModel(Setup,WeakForm,ICs,BCs,Solver,Output):
    """
    Top level class for the PBF-LB/M problem.
    Private (sub)-methods are defined in sub-classes that are
    inherited from.
    """
    def __init__(self,mesh_path: str, config: dict, bc_markers: dict, timestep: float) -> None:
        """
        The constructor for `PBFModel`.

        Args:
        mesh_path (str): the OS relative path to the mesh file
        config (dict): A dictionary of dictionaries with the following structure
            keys: all PDE fields
            values: the type of Finite Element (`element`) 
                    and degree for each field (`degree`)
        bc_markers (dict): a dictionary with integer values, enumerating the boundary
            subdomains
        timestep (float): the (for now) fixed time step for the Euler scheme
        """
        self.mesh = Mesh(meshfile=mesh_path)
        #self.mesh = UnitCubeMesh(10,10,10, hexahedral=False)
        self.config = config
        self.bc_markers = bc_markers
        self.time = 0.0
        self.dt = timestep
    
    def setup(self,outfile,**kwargs) -> None:
        self._setup_finite_element()
        self._setup_function_space()
        self._setup_functions()
        self._project_initial_conditions()
        self._create_output(filename=outfile,**kwargs)
        self._setup_bcs()
        # Set values for t = 0 to output
        self.timestep_update()
        self.write_output()
    
    def assemble(self) -> None:
        self._assemble_weak_form()

    def build_solver(self, solver_parameters: dict | None = None) -> None:
        self._assemble_problem()
        self._assemble_solver(solver_parameters)
    
    def _project_initial_conditions(self) -> None:
        # Data access must be done using `subfunctions`
        # https://www.firedrakeproject.org/demos/camassaholm.py.html
        a_s, a_l, a_g, p, u, T = self.solution.next.subfunctions

        # Temperature
        self._set_IC_T(T=T)
        # Velocity
        self._set_IC_u(u=u)
        # Phase fractions
        self._set_IC_phases(gas=a_g,solid=a_s)
    
    def timestep_update(self) -> None:
        # Data access must be done using `subfunctions`
        # https://www.firedrakeproject.org/demos/camassaholm.py.html
        self.solution.previous.assign(self.solution.next)