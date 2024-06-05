from .Mesh import RBMesh
from .FEData import RBData
from .Output import Output
from .MaterialModel import RBMaterialModel
from .InitialCondition import RBInitialConditions
from .BoundaryCondition import RBBoundaryConditions
from .Solver import PBFSolver

class RisingBubbleModel:
    def __init__(self,
                 model_parameters: dict[str,any],
                 fe_config: dict[str,dict[str,any]],
                 material_model: dict[str,dict[str,float]],
                 bc_markers: dict[str,int],
                 timestep: float, time_domain: tuple[float,float],
                 create_mixed: bool = False) -> None:
        
        
        self.mesh           = RBMesh(points=model_parameters["coords"], 
                                    n=model_parameters["grid_size"],
                                    bc_markers=bc_markers)
        self.fe_data        = RBData(mesh=self.mesh, config=fe_config,
                                     create_mixed=create_mixed)
        self.material_model = RBMaterialModel(mesh=self.mesh, fe_data=self.fe_data,
                                              material_model=material_model)
        self.time_domain    = time_domain
        self.current_time   = self.time_domain[0]
        self.dt             = timestep
        self.ics            = RBInitialConditions(params=model_parameters)
        self.bcs            = RBBoundaryConditions(fe_data=self.fe_data, 
                                                   parameters=model_parameters)
        self.output         = Output(path="output/",
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

    