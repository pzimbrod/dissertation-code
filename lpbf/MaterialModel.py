from dolfinx.fem import Constant
from dolfinx.mesh import Mesh
from dolfinx import default_scalar_type

class MaterialModel:
    def __init__(self,mesh: Mesh,parameters: dict) -> None:
        for (key, val) in parameters.items():
            setattr(self,key,Constant(mesh,default_scalar_type(val)))

        return

class MultiphaseModel:
    def __init__(self, mesh: Mesh, model: dict) -> None:
        for (key, val) in model.items():
            setattr(self,key,MaterialModel(mesh,val))