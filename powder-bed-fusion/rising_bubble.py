# %%
import numpy as np

a = 1.44e-3

model_parameters = {
    "a":            a,
    "coords":       ((-2*a,-2*a),(2*a,2*a)),
    "grid_size":    (128,128),
    "temp_top":     291.152, # K
    "temp_bottom":  290.000, # K
}


markers = {
    "leftright":    lambda x: np.isclose(x[0], -2*a) | np.isclose(x[0],2*a),
    "top":          lambda x: np.isclose(x[1], 2*a),
    "bottom":       lambda x: np.isclose(x[1], -2*a),
}

material_model = {
    "alpha1": {
        "rho":      250.0,
        "cp":       5e-5,
        "mu":       0.012,
        "kappa":    1.2e-6,
    },
    "alpha2": {
        "rho":      500.0,
        "cp":       1e-4,
        "mu":       0.024,
        "kappa":    2.4e-6
    },
}


# %%
fe_config = {

    "alpha1": {
        "element": "DG", 
        "degree": 1, 
        "type": "scalar",
        "time_scheme":  "explicit euler",
    },

    "alpha2": {
        "element": "DG", 
        "degree": 1, 
        "type": "scalar",
        "time_scheme":  "explicit euler",
        },

    "p": {
        "element": "DG", 
        "degree": 1, 
        "type": "scalar",
        "time_scheme":  "explicit euler",
        },

    "u": {
        "element": "CG", 
        "degree": 1, 
        "type": "vector",
        "time_scheme":  "explicit euler",
        },

    "T": {
        "element": "CG", 
        "degree": 1, 
        "type": "scalar",
        "time_scheme":  "explicit euler",
        },
}

# %% [markdown]
# According to [this discussion](https://fenicsproject.discourse.group/t/cannot-find-dofs-for-discontinuous-element/10900/3), discontinuous spaces do not have DoFs that live on a facet. Thus, `locate_dofs_topological` does not find any DoFs there. Must rely on `locate_dofs_geometrical` for now.

# %%
from MultiphysicsModel.RisingBubbleModel import RisingBubbleModel

model = RisingBubbleModel(
    model_parameters=model_parameters,
    material_model=material_model,
    fe_config=fe_config,
    bc_markers=markers,
    timestep=2.5e-5,
    time_domain=(0.0,0.12),
    create_mixed=True
)

# %%
model.setup()

# %%
model.solve()


