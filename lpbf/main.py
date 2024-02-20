import numpy as np
from geometry import create_geometry, check_msh_file
from mpi4py import MPI
import os
from PBFModel import PBFModel
from dolfinx import log

#========   Set this to True if you want to overwrite an existing mesh (if one is present) ==========#
# BUG: Topology computation for hex mesh never finishes, use tets for now
use_hex_mesh = False
create_new_mesh = False

current_directory = os.getcwd()
mesh_already_present = check_msh_file(current_directory)
model_rank = 0

if not mesh_already_present or create_new_mesh:
    create_geometry()

# from GEO file
markers = {
    "inlet":    23,
    "outlet":   24,
    "walls":    25,
    "bottom":   21
}

fe_config = {
    "alphas":   {"element": "DG", "degree": 1},
    # pressure-velocity uses stable Taylor-Hood pairing
    "p":        {"element": "CG", "degree": 2},
    "u":        {"element": "CG", "degree": 3},
    "T":        {"element": "CG", "degree": 1},
}

# This should be a dictionary of dictionaries
material_model = {
    "solid": {
        "rho":      7800.0,   # kg/m3
        "cp":       502.4,  # J/(kg K)
        "kappa":    14.4,   # W/(m K)
    },
}

dt = 1e-9

log.set_log_level(log.LogLevel.INFO)
model = PBFModel(mesh_path="mesh3d.msh",config=fe_config, bc_markers=markers, timestep=dt)
model.setup(material_model=material_model,filename="output/lpbf.xdmf")

model.assemble()
model.build_solver()

print("Starting solve!")
model.solve()
model.solve()
model.solve()
model.solve()
