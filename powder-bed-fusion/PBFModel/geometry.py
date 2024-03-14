import gmsh
import numpy as np
import os

"""
Create the computational mesh for the hexahedral domain with surface markers.

inlet: left wall where shielding gas flows in
outlet: right wall where shielding gas flows out
bottom: bottom wall where Dirichlet BC for temperature is applied
walls: rest of the surfaces where Neumann BCs are applied
"""
def create_geometry(markers: dict, build_hex_mesh: bool = False):
    gmsh.initialize()

    gmsh.model.add("PBF-LB/M 3D")

    # Create the bounding box
    L, B, H_0, H_1 = 2.0, 0.3, 0.3, 0.2
    
    domain_bottom = gmsh.model.occ.addBox(0,0,0,L,B,H_0)
    domain_top = gmsh.model.occ.addBox(0,0,H_0,L,B,H_1)
    domain = gmsh.model.occ.fragment([(3,domain_bottom)],[(3,domain_top)])

    gmsh.model.occ.synchronize()
    surfaces = gmsh.model.getEntities(dim=2)
    inlet_marker, outlet_marker, wall_marker, bottom_marker = 1, 3, 5, 7
    walls = []
    for surface in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0],surface[1])
        if np.allclose(com, [L/2, B, H_0+H_1/2]):
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], 
                                        markers["inlet"])
            gmsh.model.setPhysicalName(surface[0], markers["inlet"], 
                                       "Shielding gas inlet")
        elif np.allclose(com, [L/2, 0, H_0+H_1/2]):
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], 
                                        markers["outlet"])
            gmsh.model.setPhysicalName(surface[0], markers["outlet"], 
                                       "Shielding gas outlet")
        elif np.allclose(com, [L/2, B/2, 0]):
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], 
                                        markers["bottom"])
            gmsh.model.setPhysicalName(surface[0], markers["bottom"], 
                                       "Solid bottom")
        else:
            walls.append(surface[1])
    # Write the remainders to "walls"
    gmsh.model.addPhysicalGroup(2, walls, markers["walls"])
    gmsh.model.setPhysicalName(2, markers["walls"], "Walls")
    volumes = gmsh.model.getEntities(dim=3)
    gmsh.model.addPhysicalGroup(3, volumes[0])

    # True generates a Hex mesh
    if build_hex_mesh:
        NN = 30
        for c in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(c[1], NN)
        for s in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(s[1])
            gmsh.model.mesh.setRecombine(s[0], s[1])
            gmsh.model.mesh.setSmoothing(s[0], s[1], 100)
        for v in gmsh.model.getEntities(3):
            gmsh.model.mesh.setTransfiniteVolume(v[1])

    # Generate mesh
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    if not build_hex_mesh:
        n_refinements = 3
        for _ in range(0,n_refinements):
            gmsh.model.mesh.refine()
        gmsh.model.occ.synchronize()

    gmsh.write("mesh3D.msh")

def check_msh_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".msh"):
                return True
    return False