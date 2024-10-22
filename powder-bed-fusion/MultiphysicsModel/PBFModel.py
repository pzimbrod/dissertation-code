from .Mesh import PBFMesh
from .FEData import PBFData
from .MaterialModel import PBFMaterialModel
from .Output import Output
from .InitialCondition import PBFInitialConditions
from .BoundaryCondition import PBFBoundaryConditions
from .Solver import PBFSolver

class PBFModel:
    """
    The top level representation of the complete powder bed fusion model.

    ...

    Attributes
    ----------
    `mesh` : `Mesh`
        the computational domain, including information about the boundary

    `fe_data` : `FEData`
        all Finite Element specific data, including function spaces, functions and the weak form

    `material_model` : `MaterialModel`
        constants and properties that describe the problem on a physical level, as well as
        methods to compute mixture quantities

    `time_domain` : `tuple[float,float]`
        start and end time of the simulation

    `current_time` : `float`
        the current time step

    `dt` : `float`
        the (fixed) time increment

    `output` : `Output`
        the handler for creating and modifying output files
    
    `ics` : `InitialConditions`
        the methods that define the initial state of the solution variables
    
    `bcs` : `BoundaryConditions`
        the data structure holding the boundary conditions for the solution variables
    
    `solver` : `PBFSolver`
        the Nonlinear variational solver for the PBF problem

    Methods
    -------
    setup()
        After the `PBFModel` is initially created, this method assigns the necessary initial
        and boundary conditions, writes output for the first time step and sets up the Nonlinear
        variational solver.

    solve()
        Solves the problem over the entire time domain specified by `time_domain`.
    """
    def __init__(self, mesh_path: str, fe_config: dict[str,dict[str,any]],
                 material_model: dict[str,dict[str,float]],bc_markers: dict[str,int],
                 timestep: float, time_domain: tuple[float,float],
                 create_mixed: bool = False) -> None:
        """
        Parameters
        ----------
        `mesh_path` : `str`
            the exact path from the project directory where the mesh file is located

        `fe_config` : `dict[str,dict[str,any]]`
            the specification of how the Finite Element problem is supposed to be set up.
            It should have the following structure: the top level keys are the fields to be created.
            Then, for each field, the keys `element` (element type - `"CG" or "DG"`), `degree` (Finite
            Element degree - `int`) and `type` (tensor dimension - `"scalar"` or `"vector"`) must be given.
            The key "time_scheme" is optional and specifies the time stepping scheme to be used for that
            quantity. Choose between `"implicit euler"` and `"explicit euler"`.

        `material_model` : `dict[str,dict[str,float]]`
            The material parameters to be used. It should have the following structure: the top level keys
            represent the phases present in the model, they must correspond to the `alpha_*` fields in
            `fe_config`. Then, for each phase, the properties are specified via key-value pairs.

        `bc_markers` : `dict[str,int]`
            A dictionary containing the integer IDs that can be used to identify mesh boundaries.
            The `int` values of this dict are specified in the GMSH GEO file and must be set there,
            since they are read in by `dolfinx`.
        
        `time_domain` : `tuple[float,float]`
            The start and end time of the simulation
        
        `dt` : `float`
            The fixed time step increment
        
        `create_mixed` : `bool`, optional
            Whether to create the problem in a mixed space. Setting this to `True` will create a monolithic
            problem that is solved at once using a Newton method, which can take considerably longer than
            a split problem, but doesn't require any specific type of solver setup that is tailored to
            the PDE problem.
        """
        
        self.mesh           = PBFMesh(mesh_path=mesh_path, 
                                   bc_markers=bc_markers)
        self.fe_data        = PBFData(mesh=self.mesh, config=fe_config,
                                     create_mixed=create_mixed)
        self.material_model = PBFMaterialModel(mesh=self.mesh,
                                            material_model=material_model,
                                            fe_data=self.fe_data)
        self.time_domain    = time_domain
        self.current_time   = self.time_domain[0]
        self.dt             = timestep
        self.ics            = PBFInitialConditions()
        self.bcs            = PBFBoundaryConditions()
        self.output         = Output(path="pbf_output/",
                                     mesh=self.mesh,
                                     fe_data=self.fe_data)

        print("Model has Degrees of Freedom (DoFs):\n"+
              self.fe_data.count_dofs())

        return

    
    def setup(self) -> None:
        """
        After the `PBFModel` is initially created, assign the necessary initial
        and boundary conditions, write output for the first time step and set up
        the PDE weak form as well as the nonlinear solver for the problem.
        """
        self.ics.apply(fe_data=self.fe_data)
        self.output.write(fe_data=self.fe_data,
                          time=self.current_time)
        self.bcs.apply(mesh=self.mesh, fe_data=self.fe_data)
        self.fe_data.setup_weak_form(dt=self.dt,
                                     material_model=self.material_model)
        self.solver = PBFSolver(fe_data=self.fe_data, 
                             bc_data=self.bcs,
                             mesh=self.mesh)

        return

    
    def _solve_timestep(self):
        self.current_time += self.dt
        solver = self.solver.newton_solver
        if self.fe_data.is_mixed:
            u = self.fe_data.mixed_solution.current
        else:
            u = self.fe_data.solution["alpha_solid"].current

        its, is_converged = solver.solve(u=u)
        assert(is_converged)
        print(f"Nonlinear solve converged in {its} iterations.")
        self.solver.postprocess(fe_data=self.fe_data)
        self.output.write(fe_data=self.fe_data, 
                          time=self.current_time)
        self.fe_data.solution.update()

        return


    def solve(self) -> None:
        """
        Solve the nonlinear problem over all time steps.
        """
        end_time = self.time_domain[1]
        while self.current_time < end_time:
            print(f"Time: {self.current_time}")
            self._solve_timestep()
        
        print("Solve finished!")

        return

    