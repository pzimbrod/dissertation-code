from dolfinx.io import VTKFile
from .Mesh import Mesh
from .FEData import AbstractFEData

class Output:
    def __init__(self, path: str, mesh: Mesh, fe_data: AbstractFEData) -> None:
        """
        Initializes an instance of the Output class.

        Parameters:
            path (str): The path where the output files will be saved.
            mesh (Mesh): The mesh object representing the computational domain.
            fe_data (AbstractFEData): The Finite Element data object.

        Returns:
            None

        This function initializes the Output class by creating VTK files for each field in the fe_data object. 
        The files are saved in the specified path with the field name appended to the path. The mesh is written 
        to each VTK file using the write_mesh method of the VTKFile class.
        """
        self.files = {}
        functions = fe_data.solution
        for field in functions.keys():
            self.files[field] = VTKFile(
                mesh.dolfinx_mesh.comm, 
                f"{path}{field}.pvd","w")
            with self.files[field] as file:
                file.write_mesh(mesh=mesh.dolfinx_mesh)
        
        return
    

    def write(self, fe_data: AbstractFEData, time:float) -> None:
        """
        Write the solution data to the output files.

        Parameters:
            fe_data (AbstractFEData): The Finite Element data.
            time (float): The current time.

        Returns:
            None
        """
        functions = fe_data.solution
        for (field, function) in functions.items():
            with self.files[field] as file:
                if fe_data.is_mixed:
                    file.write_function(u=function.current.collapse(),t=time)
                else:
                    file.write_function(u=function.current,t=time)

        
        return

