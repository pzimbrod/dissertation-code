from dolfinx.io import (VTKFile, XDMFFile, VTXWriter)
from dolfinx.fem import (Function)
from mpi4py import MPI
from .Mesh import Mesh
from .FEData import FEData

class Output:
    def __init__(self, path: str, mesh: Mesh, fe_data: FEData) -> None:
        self.files = {}
        functions = fe_data.solution
        for field in functions.keys():
            self.files[field] = VTKFile(
                mesh.dolfinx_mesh.comm, 
                f"{path}{field}.pvd","w")
            with self.files[field] as file:
                file.write_mesh(mesh=mesh.dolfinx_mesh)
        
        return
    

    def write(self, fe_data: FEData, time:float) -> None:
        functions = fe_data.solution
        for (field, function) in functions.items():
            with self.files[field] as file:
                if fe_data.is_mixed:
                    file.write_function(u=function.current.collapse(),t=time)
                else:
                    file.write_function(u=function.current,t=time)

        
        return

