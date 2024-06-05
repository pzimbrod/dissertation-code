from dolfinx.fem import Constant, Function, Form
from .Mesh import AbstractMesh
from dolfinx import default_scalar_type
from ufl import (sqrt, exp, inner, outer, dot, grad, Identity, SpatialCoordinate)
import numpy as np

class AbstractMaterialModel:
    def __init__(self,mesh: AbstractMesh, fe_data, material_model: dict[str,float]) -> None:
        self.constants = self.__init_constants(mesh,material_model)
        self.expressions = self.__init_multiphase_expressions(fe_data)

        return


    def __init_constants(self, mesh: AbstractMesh, 
                material_model: dict[str,float]) -> dict[str,Constant]:
        constants = {}
        # Phase models
        for (phase,subdict) in material_model.items():
            constants[phase] = {}
            for (quantity, value) in subdict.items():
                constants[phase][quantity] = Constant(mesh.dolfinx_mesh,value)
        
        return constants


    def __get_multiphase_property(self, fe_data,
                            quantity: str, temporal_scheme: str) -> Form:

        phases = [fe_data.get_function(phase,temporal_scheme) for phase in self.constants.keys()]
        quantities = [self.constants[phase][quantity] for phase in self.constants.keys()]

        return np.inner(phases,quantities)

    
    def __init_multiphase_expressions(self, fe_data) -> Form:
        first_phase = next(iter(self.constants.keys()))
        phase_keys = set(self.constants[first_phase].keys())
        alpha_time_scheme = fe_data.get_time_scheme(key=first_phase)

        for phase, phase_model in self.constants.items():
            sub_dict_keys = set(phase_model.keys())
            missing_keys = phase_keys - sub_dict_keys
            extra_keys = sub_dict_keys - phase_keys
            if missing_keys or extra_keys:
                raise AssertionError(
                    f"Phase model of '{phase}' does not have the same keys as the first phase model. "
                    f"Missing keys: {missing_keys}, Extra keys: {extra_keys}"
                )

        expressions = {}
        for quantity in self.constants[first_phase].keys():
            expressions[quantity] = self.__get_multiphase_property(fe_data,
                                        quantity=quantity, 
                                        temporal_scheme=alpha_time_scheme)
        
        return expressions


    def unit_vector(self,x: Function) -> Form:
            """The normalized version of a quantity, x/|x|^2"""
            return x / sqrt(inner(x,x))
    

    def VoF_interface_gradient(self, alpha1: Function, alpha2: Function) -> Form:
        """The surface normal gradient of an interface in VoF formulation"""
        return grad(alpha1)*alpha2 - alpha1*grad(alpha2)
    

    def VoF_unit_normal(self, alpha1: Function, alpha2: Function) -> Form:
        """The unit vector normal to a phase boundary in VoF formulation"""
        grad_alpha = self.VoF_interface_gradient(alpha1,alpha2)
        return self.unit_vector(x=grad_alpha)


    def capillary_stress_tensor(self, I: Identity, sigma, alpha1: Function,
                                alpha2: Function) -> Form:
        """The capillary stress tensor in VoF formulation"""
        n = self.VoF_unit_normal(alpha1,alpha2)

        return -sigma*(I - outer(n,n))




class PBFMaterialModel(AbstractMaterialModel):
    def __init__(self,mesh: AbstractMesh, fe_data, material_model: dict[str,float]) -> None:
        super().__init__(mesh,fe_data,material_model)
        self.physical_constants = self.__init_physical_constants(material_model)
        
        return


    def __init_physical_constants(self, material_model: dict[str,float]) -> dict[str,float]:
        # Physical constants
        physical_constants = {
            "p0" : 1.013e5,        # Pa, ambient pressure
            "Lv" : 1e7,            # Latent heat
            "R" : 8.014,           # Universal gas constant
            "Tv" : 3000,           # Vaporization temperature
            "sigma" : 5.670e-8,    # Stefan Boltzmann constant [W/(m^2 K^4)]
            "epsilon" : 0.3,       # Radiative emissivity
            "T_amb" : 293.0,       # Ambient temperature [K]
            "P" : 300,             # Laser power [W]
            "alpha" : 0.4,         # Laser absorptivity
            "Rl" : 50e-6,          # Laser spot radius [m]
            "R" : 8.014,           # Universal gas constant
            "M" : 55.85,           # Molar mass of steel [g/mol]
        }
        for key in physical_constants.keys():
            if key in material_model.keys():
                physical_constants[key] = material_model[key]
        
        return physical_constants

    
    def recoil_pressure(self, T: Function) -> Form:
        """The Clausius-Clapeyron equation for recoil pressure"""
        p0 = self.physical_constants["p0"]
        Lv = self.physical_constants["Lv"]
        R  = self.physical_constants["R"]
        Tv = self.physical_constants["Tv"]
        return 0.53 * p0 * Lv/R * exp(1/Tv - 1/T)


    def heat_radiation(self, T: Function) -> Form:
        sigma   = self.physical_constants["sigma"]
        epsilon = self.physical_constants["epsilon"]
        T_amb   = self.physical_constants["T_amb"]
        return sigma * epsilon * (T**4 - T_amb**4)


    def heat_laser(self, mesh: AbstractMesh) -> Form:
        P       = self.physical_constants["P"]
        alpha   = self.physical_constants["alpha"]
        Rl      = self.physical_constants["Rl"]
        x,y,_ = SpatialCoordinate(mesh.dolfinx_mesh)

        return P * 2*alpha / (np.pi * Rl**2) * np.exp(-2*(x**2+y**2)/Rl**2)
    

    def heat_vaporization(self, T: Function) -> Form:
        R = self.physical_constants["R"]
        M = self.physical_constants["M"]

        return 0.82 * self.recoil_pressure(T)/sqrt(2*np.pi*M*R*T)




class RBMaterialModel(AbstractMaterialModel):
    def __init__(self, mesh: AbstractMesh, fe_data, material_model: dict[str, float]) -> None:
        super().__init__(mesh, fe_data, material_model)
