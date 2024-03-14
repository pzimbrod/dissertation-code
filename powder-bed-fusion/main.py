import numpy as np
from PBFModel.geometry import create_geometry, check_msh_file
from mpi4py import MPI
import os
from PBFModel.PBFModel import PBFModel
from dolfinx import log
from time import time

#========   Set this to True if you want to overwrite an existing mesh (if one is present) ==========#
# BUG: Topology computation for hex mesh never finishes, use tets for now
use_hex_mesh = False
create_new_mesh = False

current_directory = os.getcwd()
mesh_already_present = check_msh_file(current_directory)
model_rank = 0

# from GEO file
markers = {
    "inlet":    23,
    "outlet":   24,
    "walls":    25,
    "bottom":   21
}
if not mesh_already_present or create_new_mesh:
    #create_geometry(markers=markers,build_hex_mesh=use_hex_mesh)
    os.system("gmsh -3 -format msh2 lpbf/mesh3d.geo -o mesh3d.msh")

create_mixed_problem = True
fe_config = {
    "alpha_solid":  {"element": "CG", 
                     "degree": 1, 
                     "type": "scalar"},
    "alpha_liquid": {"element": "CG", 
                     "degree": 1, 
                     "type": "scalar"},
    "alpha_gas":    {"element": "CG", 
                     "degree": 1, 
                     "type": "scalar"},
    # pressure-velocity uses stable Taylor-Hood pairing
    "p":        {"element": "CG", 
                 "degree": 1, 
                 "type": "scalar"},
    "u":        {"element": "CG", 
                 "degree": 2, 
                "type": "vector"},
    "T":        {"element": "CG", 
                 "degree": 1, 
                 "type": "scalar"},
}

# This should be a dictionary of dictionaries
material_model = {
    "solid": {
        "rho":      7800.0, # kg/m3
        "cp":       502.4,  # J/(kg K)
        "kappa":    14.4,   # W/(m K)
    },
    "liquid": {
        "rho":      7800.0, # kg/m3
        "cp":       502.4,  # J/(kg K)
        "kappa":    14.4,   # W/(m K)
    },
    "gas": {
        "rho":      7800.0, # kg/m3
        "cp":       502.4,  # J/(kg K)
        "kappa":    14.4,   # W/(m K)
    },
}

time_domain = (0.0, 1.0)

dt = 1e-1

log.set_log_level(log.LogLevel.INFO)
model = PBFModel(mesh_path="mesh3d.msh",
                 fe_config=fe_config,
                 material_model=material_model,
                 bc_markers=markers, 
                 timestep=dt,
                 time_domain=time_domain,
                 create_mixed=create_mixed_problem)

model.setup()
model.solve()
