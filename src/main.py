import gmsh
import os
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import gmshio
import pyvista

from geometry import create_geometry
from materials.material_properties import *
from numerics.numerical_parameters import *

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
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, model_rank)
else:
    mesh, cell_tags, facet_tags = gmshio.read_from_msh("mesh3D.msh", MPI.COMM_WORLD, 0, gdim=3)

facet_tags.name = "Facet markers"
inlet_marker, outlet_marker, wall_marker, bottom_marker = 1, 3, 5, 7
markers = [
    inlet_marker,
    outlet_marker,
    wall_marker,
    bottom_marker
]

from function_spaces import create_fe_functions
function_space, functions = create_fe_functions(mesh=mesh, degree=degree)

from initial_condition import project_initial_conditions
project_initial_conditions(functions=functions)

from materials.material_properties import setup_constants
properties = setup_constants(mesh=mesh)

from weak_form import generate_weak_form, upwind, lax_friedrichs, HLLE
F = generate_weak_form(mesh, function_space=function_space,
                        functions=functions, flux_function=upwind, constants=properties)

prob = NonlinearProblem(F=F,u=functions, bcs=[])
