from dolfinx.io import gmshio
from mpi4py import MPI

class Mesh:
    def __init__(self, mesh_path: str, bc_markers: dict) -> None:
        self.dolfinx_mesh, self.cell_tags, self.facet_tags = gmshio.read_from_msh(
            filename=mesh_path,comm=MPI.COMM_WORLD)
        
        self.dolfinx_mesh.name = "Computational domain"
        self.cell_tags.name = "Cell markers"
        self.facet_tags.name = "Facet markers"


        # Useful for boundary conditions
        self.mesh_dim = self.dolfinx_mesh.topology.dim
        self.facet_dim = self.mesh_dim - 1

        self.dolfinx_mesh.topology.create_connectivity(self.mesh_dim,
                                                       self.facet_dim)
        self.dolfinx_mesh.topology.create_connectivity(self.facet_dim,
                                                       self.mesh_dim)
        
        self.bc_markers = bc_markers
        
        return