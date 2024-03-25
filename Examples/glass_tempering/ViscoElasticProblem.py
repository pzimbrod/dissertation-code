from dolfinx.mesh import create_interval, locate_entities_boundary
from mpi4py import MPI
from dolfinx.mesh import locate_entities_boundary
from dolfinx import fem, io, plot, nls, log
from dolfinx.io import gmshio
from dolfinx.fem import (FunctionSpace, Function, Constant, dirichletbc,
                         locate_dofs_geometrical, form, locate_dofs_topological,
                         assemble_scalar, VectorFunctionSpace, Expression)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, NonlinearProblem
from ufl import (TrialFunction, TestFunction, FiniteElement, grad, dot, inner,
                 lhs, rhs, Measure, SpatialCoordinate, FacetNormal, TensorElement, Identity)  # , ds, dx
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
import ufl
from math import factorial


class ViscoElasticProblem:
    def __init__(self, mesh_path, dt, degree=1, tensor_degree=1) -> None:
        self.mesh, cell_tags, facet_tags = gmshio.read_from_msh(
            mesh_path, MPI.COMM_WORLD, 0, gdim=1)
        self.fe = FiniteElement("P", self.mesh.ufl_cell(), degree)
        self.tfe = TensorElement("P", self.mesh.ufl_cell(), tensor_degree)
        self.fs = FunctionSpace(mesh=self.mesh, element=self.fe)
        self.tfs = FunctionSpace(mesh=self.mesh, element=self.tfe)
        self.dt = dt

        # For nonlinear problems, there is no TrialFunction
        self.T_current = Function(self.fs)
        self.T_current.name = "Temperature"
        self.v = TestFunction(self.fs)
        self.T_previous = Function(self.fs)      # previous time step

    def set_initial_condition(self, init_value: float) -> None:
        x = SpatialCoordinate(self.mesh)

        def temp_init(x):
            values = np.full(x.shape[1], init_value, dtype=ScalarType)
            return values
        self.T_previous.interpolate(temp_init)
        self.T_current.interpolate(temp_init)

    def set_dirichlet_bc(self, bc_value: float) -> None:
        return

    def write_initial_output(self, output_name: str, t: float = 0.0) -> None:
        self.xdmf = io.XDMFFile(self.mesh.comm, f"{output_name}.xdmf", "w")
        self.xdmf.write_mesh(self.mesh)
        self.xdmf.write_function(self.T_current, t)

    def setup_weak_form(self, parameters: dict) -> None:
        return

    def setup_solver(self) -> None:
        return

    def solve(self, t) -> None:
        return

    def finalize(self) -> None:
        self.xdmf.close()


class ViscoElasticModel:
    def __init__(self, prob: ViscoElasticProblem, parameters: dict) -> None:
        self.mesh = prob.mesh
        dim = self.mesh.topology.dim
        # Identity tensor
        self.I = ufl.Identity(dim)
        # Activation energy [J/mol]
        self.H = Constant(self.mesh, parameters["H"])
        # Universal gas constant [J/(mol K)]
        self.Rg = Constant(self.mesh, parameters["Rg"])
        # Base temperature [K]
        self.Tb = Constant(self.mesh, parameters["Tb"])
        # Solid thermal expansion coefficient [K^-1]
        self.alpha_solid = Constant(self.mesh, parameters["alpha_solid"])
        # Liquid thermal expansion coefficient [K^-1]
        self.alpha_liquid = Constant(self.mesh, parameters["alpha_liquid"])
        # weighting coefficient for temperature and structural energies, c.f. Nielsen et al. eq. 8
        self.chi = 0.5
        self.dt = prob.dt
        self.Tf_current = Function(prob.fs)
        self.Tf_previous = Function(prob.fs)
        
        self.m_n_tableau = np.array([
            5.523e-2,
            8.205e-2,
            1.215e-1,
            2.286e-1,
            2.860e-1,
            2.265e-1,
        ])
        self.lambda_m_n_tableau = np.array([
            5.965e-4,
            1.077e-2,
            1.362e-1,
            1.505e-1,
            6.747e+0,
            2.963e+1,
        ])
        self.g_n_tableau = np.array([
            1.585,
            2.354,
            3.486,
            6.558,
            8.205,
            6.498,
        ])
        self.lambda_g_n_tableau = np.array([
            6.658e-5,
            1.197e-3,
            1.514e-2,
            1.672e-1,
            7.497e-1,
            3.292e+0
        ])
        self.k_n_tableau = np.array([
            7.588e-1,
            7.650e-1,
            9.806e-1,
            7.301e+0,
            1.347e+1,
            1.900e+1,
            7.50e+0,
        ])
        self.lambda_k_n_tableau = np.array([
            5.009e-5,
            9.945e-4,
            2.022e-3,
            1.925e-2,
            1.199e-1,
            2.033e+0,
            1.000e+30,  # instead of Inf
        ])
        
        # Intermediate functions
        # Fictive temperature
        self.Tf_partial_current = [Function(prob.fs) for _ in range(0,self.m_n_tableau.size)]
        self.Tf_partial_previous = [Function(prob.fs) for _ in range(0,self.m_n_tableau.size)]
        # Deviatoric stress (tensor)
        self.ds_partial_current = [Function(prob.tfs) for _ in range(0,self.g_n_tableau.size)]
        self.ds_partial_previous = [Function(prob.tfs) for _ in range(0,self.g_n_tableau.size)]
        # Hydrostatic stress (scalar)
        self.dsigma_partial_current = [Function(prob.fs) for _ in range(0,self.m_n_tableau.size)]
        self.dsigma_partial_previous = [Function(prob.fs) for _ in range(0,self.m_n_tableau.size)]
        # Total stress (tensor)
        self.stress_tensor = Function(prob.tfs)        

    def compute_Tf_current(self,T_current,dt):
        self.Tf_current.interpolate(self._Tf_current(T_current,dt))
        return self.Tf_current
    
    def _Tf_partial_current(self,T_current,dt,phi_v):
        """
        Update current values for partial fictive temperature based on previous values.
        C.f. Nielsen et al., eq. 24
        """
        self.Tf_partial_current = (self.lambda_m_n_tableau * self.Tf_partial_previous + T_current * dt * phi_v) / \
                            (self.lambda_m_n_tableau + dt * phi_v)
        return self.Tf_partial_current

    
    def _Tf_current(self,T_current,dt):
        """
        Perform weighted summation of all partial fictive temperature values.
        C.f. Nielsen et al., eq. 26
        """
        # Reset for accumulation
        self.Tf_current = np.dot(self._Tf_partial_current(T_current,dt,self._phi_v(T_current)),self.m_n_tableau)
        return self.Tf_current

    def _phi_v(self, T_current):
        """
        The shift function, c.f. Nielsen et al., eq. 25
        """
        return ufl.exp(
            self.H / self.Rg * (
                1 / self.Tb -
                self.chi / T_current -
                (1 - self.chi) / self.Tf_previous
            )
        )

    def _eps_th(self):
        """
        Thermal strain tensor, c.f. Nielsen et al., eq. 9
        """
        return self.I * (
            self.alpha_solid * (self.Tf_current - self.Tf_previous)
            - (self.alpha_liquid - self.alpha_solid) * (self.Tf_current - self.Tf_previous)
            )
    
    def _strain_increment_tensor(self):
        """
        The total strain tensor. In absence of mechanical loads, this is trivially given.
        C.f. Nielsen et al., eq. 28
        """
        return -self._eps_th()

    def _eps_dev(self):
        """
        The Deviatoric strain increment tensor, c.f. Nielsen et al., eq. 29
        """
        eps = self._strain_increment_tensor()
        return eps - 1. / 3 * ufl.tr(eps) * self.I
    
    def _phi(self,T):
        """
        The shift function, c.f. Nielsen et al., eq. 5
        """
        return ufl.exp(self.H / self.Rg * (1.0 / self.Tb - 1.0 / T))

    def _dxi(self,T_current,T_previous):
        """
        The shifted time, c.f. Nielsen et al., eq. 19
        """
        return self.dt / 2.0 * (self._phi(T_current) - self._phi(T_previous))
    
    def _taylor_exponential(self,T_current,T_previous,which_lambda):
        """
        The stability correction for dxi -> 0, replaces the exponential
        by a three parts taylor expansion, c.f. Nielsen et al., eq. 20
        """
        expr = 1.0
        dxi = self._dxi(T_current,T_previous)
        if which_lambda == "g":
            lam = self.lambda_g_n_tableau
        elif which_lambda == "k":
            lam = self.lambda_k_n_tableau
        for k in range(0,3):
            expr -= 1.0 / factorial(k) * (- dxi / lam)**k
        return expr
    
    def _ds_partial_previous(self,T_current,T_previous):
        """
        The partial deviatoric stress increment at previous time, c.f. Nielsen et al., eq. 15a
        """
        self.ds_partial_previous = 2.0 * self.g_n_tableau * self._eps_dev()/self._dxi(T_current,T_previous) * self.lambda_g_n_tableau \
                                    * self._taylor_exponential(T_current,T_previous,"g")
        return self.ds_partial_previous
    
    def _dsigma_partial_previous(self,T_current,T_previous):
        """
        The partial hydrostatic stress increment at previous time, c.f. Nielsen et al., eq. 15b
        """
        self.dsigma_partial_previous = self.k_n_tableau * self._strain_increment_tensor()/self._dxi(T_current,T_previous) * self.lambda_k_n_tableau \
                                    * self._taylor_exponential(T_current,T_previous,"k")
        return self.dsigma_partial_previous

    def _ds_partial_current(self,T_current,T_previous):
        """
        The partial deviatoric stress increment at current time, c.f. Nielsen et al., eq. 16a
        """
        self.ds_partial_current = self._ds_partial_previous(T_current,T_previous) * self._taylor_exponential(T_current,T_previous,"g")
        return self.ds_partial_current

    def _dsigma_partial_current(self,T_current,T_previous):
        """
        The partial hydrostatic stress increment at current time, c.f. Nielsen et al., eq. 16b
        """
        self.dsigma_partial_current = self._dsigma_partial_previous(T_current,T_previous) * self._taylor_exponential(T_current,T_previous,"k")
        return self.dsigma_partial_current
    
    def compute_stress_tensor(self,T_current,T_previous):
        """
        The total stress tensor at current time, c.f. Nielsen et al., eq. 18
        """
        self.stress_tensor.interpolate(sum(self._ds_partial_current(T_current,T_previous) + self._ds_partial_previous(T_current,T_previous)) \
                            + self.I * sum(self._dsigma_partial_current(T_current,T_previous) + self._dsigma_partial_previous(T_current,T_previous)))
        return self.stress_tensor

    def _M(self,scaled_time):
        """
        The response function M(xi), c.f. Nielsen et al., Eq. 23   
        args:
            scaled_time: The scaled time xi
        returns:
            expr: A UFL expression representing the response function
        """
        # Initialize value to loop on
        expr = 0.0
        for (m_n,lambda_m_n) in zip(self.m_n_tableau,self.lambda_m_n_tableau):
            expr += m_n * ufl.exp(- scaled_time / lambda_m_n)
        return expr
    
    def _G(self, t: float, g_n_tableau: np.ndarray,lambda_g_n_tableau: np.ndarray):
        """
        The shear relaxation modulus, c.f. Nielsen et al., eq. 11
        """
        g = 0.0
        for (g_n,lambda_g_n) in zip(g_n_tableau,lambda_g_n_tableau):
            g += g_n * ufl.exp( - t/lambda_g_n)
        return g

    
    def _K(self, t: float, k_n_tableau: np.ndarray,lambda_k_n_tableau: np.ndarray):
        """
        The bulk relaxation modulus, c.f. Nielsen et al., eq. 11
        """
        k = 0.0
        for (k_n,lambda_k_n) in zip(k_n_tableau,lambda_k_n_tableau):
            k += k_n * ufl.exp( - t/lambda_k_n)
        return k