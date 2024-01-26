import numpy as np
from geometry import create_geometry, check_msh_file
from mpi4py import MPI
import os
from PBFModel import PBFModel

#========   Set this to True if you want to overwrite an existing mesh (if one is present) ==========#
create_new_mesh = False

current_directory = os.getcwd()
mesh_already_present = check_msh_file(current_directory)
model_rank = 0

if not mesh_already_present or create_new_mesh:
    create_geometry()

inlet_marker, outlet_marker, wall_marker, bottom_marker = 1, 3, 5, 7
markers = {
    "inlet":    inlet_marker,
    "outlet":   outlet_marker,
    "walls":    wall_marker,
    "bottom":   bottom_marker
}

config = {
    "alphas":   {"element": "DG", "degree": 1},
    "p":        {"element": "DG", "degree": 1},
    "u":        {"element": "CG", "degree": 1},
    "T":        {"element": "CG", "degree": 1},
}

dt = 1e-3

model = PBFModel(mesh_path="mesh3d.msh",config=config, bc_markers=markers, timestep=dt)
model.setup(outfile="output/lpbf.pvd")

model.assemble()
model.build_solver()

print("Starting solve!")
model.solve()