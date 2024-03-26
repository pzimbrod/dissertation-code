from dolfinx.fem import Constant, Function, Form
from .Mesh import Mesh
from dolfinx import default_scalar_type

class MaterialModel:
    def __init__(self,mesh: Mesh, fe_data, material_model: dict[str,float]) -> None:
        self.constants = self.__init_constants(mesh,material_model)
        self.expressions = self.__init_multiphase_expressions(fe_data)
        
        return


    def __init_constants(self, mesh: Mesh, 
                material_model: dict[str,float]) -> dict[str,Constant]:
            
        constants = {}
        for (phase,subdict) in material_model.items():
            constants[phase] = {}
            for (quantity, value) in subdict.items():
                constants[phase][quantity] = Constant(mesh.dolfinx_mesh,value)

        return constants


    def __get_multiphase_property(self, fe_data,
                            quantity: str, temporal_scheme: str) -> Form:
        a_solid = fe_data.get_function("alpha_solid", temporal_scheme)
        a_liquid = fe_data.get_function("alpha_liquid", temporal_scheme)
        a_gas = fe_data.get_function("alpha_gas", temporal_scheme)
        qty_solid   = self.constants["solid"][quantity]
        qty_liquid  = self.constants["liquid"][quantity]
        qty_gas     = self.constants["gas"][quantity]

        return a_solid*qty_solid + a_liquid*qty_liquid + a_gas*qty_gas

    
    def __init_multiphase_expressions(self, fe_data) -> Form:
        alpha_time_scheme = fe_data.get_time_scheme(key="alpha_solid")
        expressions = {}
        for key in self.constants["solid"].keys():
            assert (key in self.constants["liquid"] and  key in self.constants["gas"]), \
                        f"KeyError: phase key {key} must be present for all phases"

            expressions[key] = self.__get_multiphase_property(fe_data,
                                        quantity=key, 
                                        temporal_scheme=alpha_time_scheme)
        
        return expressions