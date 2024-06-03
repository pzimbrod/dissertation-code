from dolfinx.io import gmshio
from dolfinx.mesh import Mesh, create_rectangle
from mpi4py import MPI
from numpy.typing import ArrayLike
from types import FunctionType

class AbstractMesh:
    def __init__(self, mesh: Mesh, bc_markers: any) -> None:
        self.dolfinx_mesh = mesh
        self.dolfinx_mesh.name = "Computational domain"
        
        # Useful for boundary conditions
        self.cell_dim = self.dolfinx_mesh.topology.dim
        self.facet_dim = self.cell_dim - 1

        self.dolfinx_mesh.topology.create_connectivity(self.cell_dim,
                                                       self.facet_dim)
        self.dolfinx_mesh.topology.create_connectivity(self.facet_dim,
                                                       self.cell_dim)

        self.bc_markers = bc_markers


class RBMesh(AbstractMesh):
    def __init__(self, points: ArrayLike, n: ArrayLike, bc_markers: dict[str,FunctionType]) -> None:
        mesh = create_rectangle(comm=MPI.COMM_WORLD, points=points, n=n)

        super().__init__(mesh, bc_markers)

        return


class PBFMesh(AbstractMesh):
    """
    The top level representation of the complete powder bed fusion model.

    ...

    Attributes
    ----------
    `dolfinx_mesh` : `dolfinx.mesh`
        the mesh representation that `dolfinx` uses to represent the Finite Element mesh

    `cell_tags` : `dolfinx.mesh.MeshTags`
        if created, a `dolfinx` object representing tags for physical cells in `gmsh`

    `facet_tags` : `dolfinx.mesh.MeshTags`
        a `dolfinx` object representing tags for physical facets (in this case, surfaces) in `gmsh`. 
        Necessary to assign functions to the mesh boundary

    `cell_dim` : `int`
        the mesh dimensionality (3)

    `facet_dim` : `int`
        the facet dimensionality

    `dt` : `float`
        the (fixed) time increment

    `output` : `Output`
        the handler for creating and modifying output files
    """
    def __init__(self, mesh_path: str, bc_markers: dict[str,int]) -> None:
        """
        Parameters
        ----------
        `mesh_path` : `str`
            the exact path from the project directory where the mesh file is located

        `bc_markers` : `dict[str,int]`
            A dictionary containing the integer IDs that can be used to identify mesh boundaries.
            The `int` values of this dict are specified in the GMSH GEO file and must be set there,
            since they are read in by `dolfinx`.
        """
        dolfinx_mesh, cell_tags, facet_tags = gmshio.read_from_msh(
            filename=mesh_path,comm=MPI.COMM_WORLD)

        super().__init__(mesh=dolfinx_mesh, bc_markers=bc_markers)
        
        self.cell_tags = cell_tags
        self.cell_tags.name = "Cell markers"
        self.facet_tags = facet_tags
        self.facet_tags.name = "Facet markers"

        return