#----------------------
# simulation parameters

t = 0.0001  # simulation time

#----------------------
# gaussian parameters

sigma = 0.01   # std
x_0 = 0.3    # starting position 
k_0 = 2e3      # initial momentum

# boundary conditions

boundary = 'periodic'

#-----------------------
# potential parameters

# 'flat', 'barrier', 'harmonic'

potential = 'flat'

# potential barrier / well

b = 0.001    # half width
h = 1e7    # height

# harmonic potential 

a = 5e7    # aperture

# ----------------------
# plotting parameters

freq = 10



