alpha = 0.025  # Technological growth rate
beta = 0.02  # Population growth rate
delta = 0.01  # Deprecation rate
# Phillips Curve Parameters from Keen (1995)
phi0 = 0.04 / (1 - 0.04 ** 2)
phi1 = 0.04 ** 3 / (1 - 0.04 ** 2)
# Investment Rate Parameters from Grasselli (2012)
kappa0 = -0.0065
kappa1 = math.exp(-5)
kappa2 = 20
r = 0.03  # Real interest rate
v = 3  # Capital to output ratio