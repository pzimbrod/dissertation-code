import gmsh
from dolfinx import log
from ThermalProblem import ThermalProblem
from ViscoElasticProblem import ViscoElasticProblem, ViscoElasticModel

# Time domain
t_start = 0.0
t_end = 1.0
n_steps = 101
dt = (t_end - t_start) / n_steps
t = t_start

# Triangulation and Finite element
mesh_path = "Examples/glass_tempering/mesh1d.msh"
def create_geometry(path: str):
    gmsh.initialize()
    gmsh.model.add("Glass 1D mesh")

    resolution_fine = 0.01
    resolution_mid = 1.0
    resolution_coarse = 3.0
    gmsh.model.occ.addPoint(0.0,0.0,0.0,resolution_fine,0)
    gmsh.model.occ.addPoint(5.0,0.0,0.0,resolution_mid,1)
    gmsh.model.occ.addPoint(25.0,0.0,0.0,resolution_coarse,2)
    gmsh.model.occ.addPoint(45.0,0.0,0.0,resolution_mid,3)
    gmsh.model.occ.addPoint(50.0,0.0,0.0,resolution_fine,4)

    gmsh.model.occ.addLine(0,1,0)
    gmsh.model.occ.addLine(1,2,1)
    gmsh.model.occ.addLine(2,3,2)
    gmsh.model.occ.addLine(3,4,3)

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(1, [0,1,2,3], 0)
    gmsh.model.setPhysicalName(1, 0, "cells")

    # Generate mesh
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(1)
    gmsh.write(path)

create_geometry(path=mesh_path)

poly_degree = 1

model_params = {
    "f": 0.0,
    "epsilon": 0.93,
    "sigma": 5.670e-8,
    "T_0": 600.0,
    "alpha": 0.0005
}

visco_params = {
    "H": 22.380e3,
    "Tb": 779.9e0,
    "Rg": 8.314,
    "alpha_solid": 9.10e-6,
    "alpha_liquid": 25.10e-6,
}

thermal_prob = ThermalProblem(mesh_path=mesh_path,dt=dt,degree=poly_degree)
#mech_prob = ViscoElasticProblem(mesh_path=mesh_path,dt=dt,degree=poly_degree,tensor_degree=poly_degree)
#visco_model = ViscoElasticModel(prob=mech_prob,parameters=visco_params)


initial_temp = 873.0
thermal_prob.set_initial_condition(initial_temp)

thermal_prob.setup_weak_form(parameters=model_params)
thermal_prob.write_initial_output(output_name="glass_tempering")

thermal_prob.setup_solver()

log.set_log_level(log.LogLevel.INFO)

for i in range(n_steps):
    t += dt
    thermal_prob.solve(t=t)

thermal_prob.finalize()