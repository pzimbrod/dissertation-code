from dolfinx.fem.petsc import NonlinearProblem
from dolfinx import nls
from petsc4py import PETSc

def setup_solver(mesh,F,u,bcs):
    prob = NonlinearProblem(F=F,u=u, bcs=bcs)
    solver = nls.petsc.NewtonSolver(mesh.comm, prob)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-12
    solver.report = True

    """ ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "gamg"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions() """
    return solver