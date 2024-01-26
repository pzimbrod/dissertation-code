import gmsh
from mpi4py import MPI
import numpy as np
import os

"""
Create the computational mesh for the hexahedral domain with surface markers.

inlet: left wall where shielding gas flows in
outlet: right wall where shielding gas flows out
bottom: bottom wall where Dirichlet BC for temperature is applied
walls: rest of the surfaces where Neumann BCs are applied
"""
def create_geometry():
    gmsh.initialize()

    gmsh.model.add("PBF-LB/M 3D")

    # Create the bounding box
    L, B, H = 2.0, 0.3, 0.5
    domain = gmsh.model.occ.addBox(0,0,0,L,B,H)

    gmsh.model.occ.synchronize()
    surfaces = gmsh.model.getEntities(dim=2)
    inlet_marker, outlet_marker, wall_marker, bottom_marker = 1, 3, 5, 7
    walls = []
    for surface in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0],surface[1])
        if np.allclose(com, [L/2, B, H/2]):
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], inlet_marker)
            inlet = surface[1]
            gmsh.model.setPhysicalName(surface[0], inlet_marker, "Shielding gas inlet")
        elif np.allclose(com, [L, 0, H/2]):
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], outlet_marker)
            gmsh.model.setPhysicalName(surface[0], outlet_marker, "Shielding gas outlet")
        elif np.allclose(com, [L/2, B/2, 0]):
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], bottom_marker)
            gmsh.model.setPhysicalName(surface[0], outlet_marker, "Solid bottom")
        else:
            walls.append(surface[1])
    # Write the remainders to "walls"
    gmsh.model.addPhysicalGroup(2, walls, wall_marker)
    gmsh.model.setPhysicalName(2, wall_marker, "Walls")
    volumes = gmsh.model.getEntities(dim=3)
    gmsh.model.addPhysicalGroup(3, volumes[0])

    # True generates a Hex mesh
    transfinite = True
    if transfinite:
        NN = 30
        for c in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(c[1], NN)
        for s in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(s[1])
            gmsh.model.mesh.setRecombine(s[0], s[1])
            gmsh.model.mesh.setSmoothing(s[0], s[1], 100)
        gmsh.model.mesh.setTransfiniteVolume(domain)

    # Generate mesh
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    if not transfinite:
        n_refinements = 3
        for _ in range(0,n_refinements):
            gmsh.model.mesh.refine()

    gmsh.write("mesh3D.msh")

def check_msh_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".msh"):
                return True
    return False