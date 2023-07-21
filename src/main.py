import gmsh
import os
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import (Constant, Function, FunctionSpace, 
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector, 
                               create_vector, create_matrix, set_bc)
from dolfinx.graph import create_adjacencylist
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)
from dolfinx.mesh import create_mesh, meshtags_from_entities
import pyvista

from ufl import (FacetNormal, FiniteElement, Identity, Measure, TestFunction, TrialFunction, VectorElement,
                 MixedElement,as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, variable)

from geometry import create_geometry
from material_properties import *
from numerical_parameters import *

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
testFunctions, trialFunctions, functions, fs = create_fe_functions(mesh=mesh, degree=degree)

from material_properties import setup_constants
properties = setup_constants(mesh=mesh)

from weak_form import generate_weak_form, upwind, lax_friedrichs, HLLE
A, l = generate_weak_form(mesh, trialFunctions=trialFunctions, testFunctions=testFunctions,
                        functions=functions, flux_function=upwind)