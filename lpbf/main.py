from firedrake import H1
import numpy as np
from geometry import create_geometry
from mpi4py import MPI
import os
from PBFModel import PBFModel

def check_msh_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".msh"):
                return True
    return False

#========   Set this to True if you want to overwrite an existing mesh (if one is present) ==========#
create_new_mesh = False

current_directory = os.getcwd()
mesh_already_present = check_msh_file(current_directory)
model_rank = 0

if not mesh_already_present or create_new_mesh:
    create_geometry()

inlet_marker, outlet_marker, wall_marker, bottom_marker = 1, 3, 5, 7
markers = [
    inlet_marker,
    outlet_marker,
    wall_marker,
    bottom_marker
]

degrees = {
    "alphas": 3,
    "p"     : 1,
    "u"     : 3,
    "T"     : 3,
}

model = PBFModel(mesh_path="mesh3d.msh",degrees=degrees)
model.setup("output/lpbf.pvd")
model.project_initial_conditions()
model.write_output()