from .Mesh import Mesh
from .FEData import FEData
from .MaterialModel import MaterialModel
from dolfinx.fem import Function
import numpy as np
from .Output import Output
from .InitialCondition import InitialConditions
from .BoundaryCondition import BoundaryConditions
from .Solver import Solver

class PBFModel:
    def __init__(self, mesh_path: str, fe_config: dict[dict[str,any]],
                 material_model: dict[str,float],bc_markers: dict[str,int],
                 timestep: float, time_domain: tuple,
                 create_mixed: bool = False) -> None:
        
        self.mesh           = Mesh(mesh_path=mesh_path, 
                                   bc_markers=bc_markers)
        self.fe_data      = FEData(mesh=self.mesh, config=fe_config,
                                     create_mixed=create_mixed)
        self.material_model = MaterialModel(mesh=self.mesh,
                                            material_model=material_model)
        
        self.time_domain    = time_domain
        self.current_time   = self.time_domain[0]
        self.dt             = timestep

        self.output         = Output(path="output/",
                                     mesh=self.mesh,
                                     fe_data=self.fe_data)

        return

    
    def setup(self) -> None:

        self.ics = InitialConditions()
        self.ics.apply(fe_data=self.fe_data)

        self.output.write(fe_data=self.fe_data,
                          time=self.current_time)

        self.bcs = BoundaryConditions()
        self.bcs.apply(mesh=self.mesh, fe_data=self.fe_data)
        
        self.fe_data.setup_weak_form(dt=self.dt)

        self.solver = Solver(fe_data=self.fe_data, 
                             bc_data=self.bcs,
                             mesh=self.mesh)

        return

    
    def _solve_timestep(self):
        self.current_time += self.dt
        solver = self.solver.newton_solver
        u = self.fe_data.solution["T"].current

        its, is_converged = solver.solve(u=u)
        assert(is_converged)
        print(f"Nonlinear solve converged in {its} iterations.")
        self.output.write(fe_data=self.fe_data, 
                          time=self.current_time)
        self.fe_data.solution.update()

        return


    def solve(self) -> None:
        end_time = self.time_domain[1]
        while self.current_time < end_time:
            print(f"Time: {self.current_time}")
            self._solve_timestep()
        
        print("Solve finished!")

        return

    