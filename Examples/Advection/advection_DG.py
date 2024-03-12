from firedrake import *
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from firedrake.petsc import PETSc
import numpy as np
import scipy
import time

# Geometry
PETSc.Sys.Print('setting up mesh across %d processes' % COMM_WORLD.size)
n = 384         # Number of cells
L = 5.          # Domain length
mesh = PeriodicSquareMesh(n,n,L,quadrilateral=True)
PETSc.Sys.Print('  rank %d owns %d elements and can access %d vertices' \
                % (mesh.comm.rank, mesh.num_cells(), mesh.num_vertices()),
                comm=COMM_SELF)

# Function Spaces
degree = 0
fe = FiniteElement("DQ", quadrilateral, degree, variant="spectral")
V = FunctionSpace(mesh, fe)        # Scalar transport quantity
V0 = FunctionSpace(mesh, "DQ", 0)       # Helper for the IC
W = VectorFunctionSpace(mesh, "DQ", 0)  # Velocity

# Project constant velocity into function space
velocity = [1.,1.]
x, y = SpatialCoordinate(mesh)
vel = Function(W).interpolate(as_vector(velocity))

# Initial condition
x_l = y_l = 2.
x_r = y_r = 3.
IC = conditional(And(And(x_l <= x, x <= x_r),And(y_l <= y, y <= y_r)), 1., 0.)
phi0 = Function(V).interpolate(IC)
phi = Function(V,name="phi").interpolate(phi0)
# For evaluating the L2 error, we need the IC field in a separate variable
phi_initial = Function(V).assign(phi)

# Store phi values at time steps
phi_vals = []

# Time domain and time stepping
T = 5.
# CFL = (u_max * dt) / (dx * (2p+1))  <=> dt = CFL * (dx*(2p+1))/vel_max
vel_max = np.sqrt(np.sum(velocity))
CFL = 0.2 * 1./(2*degree+1)
h = L / n
dt = CFL * h/vel_max
dt_ufl = Constant(dt)
t = 0.0

# Specify the mass matrix u_t \int_\Omega v phi dx
test = TestFunction(V)
dphi = TrialFunction(V)
bilinearForm = test * dphi * dx

def print_sparsity(form):
    A = assemble(form)
    mi, mj, mv = A.petscmat.getValuesCSR()
    Msp = scipy.sparse.csr_matrix((mv,mj,mi))
    plt.spy(Msp)
    return plt

# Define \vec{u} * \vec{n}
n = FacetNormal(mesh)
v_max = max(max(velocity),0)
v_min = min(min(velocity),0)

flux_function = "upwind"
if flux_function == "upwind":
    vel_n = 0.5*(dot(vel, n) + abs(dot(vel, n)))
    flux = jump(vel_n*phi)
elif flux_function == "Lax Friedrichs":
    flux = dot(avg(vel*phi),n('+')) + 0.5*v_max*jump(phi)
elif flux_function == "HLLE":
    flux = v_max/(v_max-v_min) * dot(vel('+'),n('+'))*phi('+') - v_min * dot(vel('-'),n('+'))*phi('-') \
            - v_max * v_min * jump(phi)

# Assemble the linearform that represents the spatially discretized system
if degree == 0:
    linearForm = dt_ufl *(- jump(test)*flux*dS)
else:
    linearForm = dt_ufl *(phi*div(test*vel)*dx - jump(test)*flux*dS)

phi1 = Function(V); phi2 = Function(V)
stage2 = replace(linearForm, {phi: phi1}); stage3 = replace(linearForm, {phi: phi2})

phi_tmp = Function(V)

# Project to a zero degree discontinuous space for visualisation purposes
outfile = File("output/output.pvd", target_degree=1, target_continuity=H1)
outfile.write(phi)

# Matrix free
params = {"mat_type": "matfree","ksp_type": "cg","ksp_monitor": None,"pc_type": "none"}
#params = {"mat_type": "matfree","ksp_type": "cg","ksp_monitor": None,"pc_type": "python","pc_python_type": "firedrake.AssembledPC","assembled_pc_type": "ilu"}
#params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob1 = LinearVariationalProblem(bilinearForm, linearForm, phi_tmp)
solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
prob2 = LinearVariationalProblem(bilinearForm, stage2, phi_tmp)
solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
prob3 = LinearVariationalProblem(bilinearForm, stage3, phi_tmp)
solv3 = LinearVariationalSolver(prob3, solver_parameters=params)

A = assemble(bilinearForm,mat_type='matfree')
b1 = assemble(linearForm,mat_type='matfree')
b2 = assemble(stage2,mat_type='matfree')
b3 = assemble(stage3,mat_type='matfree')

step = 0
output_step = 0.100

file = open("solve.csv","w")

PETSc.Sys.Print('solving problem ...')
while t <= T - 0.5*dt:
    start_time = time.time()
    solv1.solve()
    b1 = assemble(linearForm,mat_type='matfree')
    #solve(A, phi_tmp, b1, solver_parameters=params)
    #limiter.apply(phi)
    phi1.assign(phi + phi_tmp)

    solv2.solve()
    b2 = assemble(stage2,mat_type='matfree')
    #solve(A, phi_tmp, b2, solver_parameters=params)
    #limiter.apply(phi1)
    phi2.assign(0.75*phi + 0.25*(phi1 + phi_tmp))

    solv3.solve()
    b3 = assemble(stage3,mat_type='matfree')
    #solve(A, phi_tmp, b3, solver_parameters=params)
    #limiter.apply(phi2)
    phi.assign((1.0/3.0)*phi + (2.0/3.0)*(phi2 + phi_tmp))

    step += 1
    t += dt

    print(time.time()-start_time,file=file)

    is_savetime = np.round(np.round(t,3) % output_step,3)

    if (is_savetime == output_step) or (is_savetime == 0.0):
        #phi_vals.append(phi.copy(deepcopy=True))
        #outfile.write(phi, time=t)
        PETSc.Sys.Print("t=", t)

# last time step t = T
t -= dt
dt = T-t
dt_ufl = Constant(dt)
solv1.solve()
b1 = assemble(linearForm)
#solve(A, phi_tmp, b1, solver_parameters=params)
#limiter.apply(phi)
phi1.assign(phi + phi_tmp)

solv2.solve()
b2 = assemble(stage2)
#solve(A, phi_tmp, b2, solver_parameters=params)
#limiter.apply(phi1)
phi2.assign(0.75*phi + 0.25*(phi1 + phi_tmp))

solv3.solve()
b3 = assemble(stage3)
#solve(A, phi_tmp, b3, solver_parameters=params)
#limiter.apply(phi2)
phi.assign((1.0/3.0)*phi + (2.0/3.0)*(phi2 + phi_tmp))

t += dt

outfile.write(phi, time=t)
PETSc.Sys.Print("t=", t)


L2_err = sqrt(assemble(dot(phi - phi_initial,phi - phi_initial)*dx))
L2_init = sqrt(assemble(dot(phi_initial,phi_initial)*dx))
#PETSc.Sys.Print(L2_err/L2_init)
PETSc.Sys.Print(f'L2 error: {L2_err}')

#= Output the relevant citations for this program =#
#Citations.print_at_exit()
