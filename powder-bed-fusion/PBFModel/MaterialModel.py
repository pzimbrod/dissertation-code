from dolfinx.fem import Constant, Function
from .Mesh import Mesh
from dolfinx import default_scalar_type

class MaterialModel:
    def __init__(self,mesh: Mesh,material_model: dict) -> None:
        self.constants = {}
        for (phase,subdict) in material_model.items():
            self.constants[phase] = {}
            for (quantity, value) in subdict.items():
                self.constants[phase][quantity] = Constant(mesh.dolfinx_mesh,value)
        
        return
    
    def get_property(self,fn: Function, quantity: str):
        a_solid, a_liquid, a_gas, _, _, _ = fn.split()
        qty_solid   = self.constants["solid"][quantity]
        qty_liquid  = self.constants["liquid"][quantity]
        qty_gas     = self.constants["gas"][quantity]

        return a_solid*qty_solid + a_liquid*qty_liquid + a_gas*qty_gas