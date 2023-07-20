#### Constants ###
rho = 7800      # Density [kg/m^3]
eta = 0.1       # Viscosity [Pa s]
kappa = 0.1     # Thermal conductivity [W/(m K)]

#### Variables ####
def sigma(T):
    return 1.44 - T * 2e-4