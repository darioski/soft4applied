#----------------------
# simulation parameters

t = 1e-4 # simulation time

#----------------------
# gaussian parameters

sigma = 0.01   # std
x_0 = 0.05  # starting position 

# Nyquist frequency = 3127
k_0 = 3217     # initial momentum

# boundary conditions

boundary = 'periodic'

#-----------------------
# potential parameters

# 'flat', 'barrier', 'harmonic', 'delta'

potential = 'flat'

# potential barrier / well

b = 0.1   # half width
h = 1e7    # height

# harmonic potential 

a = 5e7    # aperture

# delta potential

alpha = 1e6    # height

# ----------------------
# plotting parameters

# 'gif', 'html', 'mp4'
file_format = 'html'  # file format

play_speed = 1   # animation speed multiplier

# video duration (seconds) = t * 1e7 * 2e-2 / speed



