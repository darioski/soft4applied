#----------------------
# simulation parameters

t = 0.001  # simulation time

#----------------------
# gaussian parameters

sigma = 0.02 # std
x_0 = 0.5    # starting position 
k_0 = 500   # initial momentum

# boundary conditions

boundary = 'periodic'   # 'periodic'

#-----------------------
# potential parameters

potential = 'barrier'   # 'flat', 'barrier', 'harmonic', 'delta'

# potential barrier / well

b = 0.2 # half width 
h = 1e7  # height: h < 0 = potential well

# harmonic potential 

a = 5e7    # aperture

# delta potential

alpha = 1e6    # height

# ----------------------
# plotting parameters

# file format
file_format = 'html'  # 'gif', 'html', 'mp4'

# animation speed multiplier

play_speed = 30 # video duration (seconds) = t * 2e5 / speed