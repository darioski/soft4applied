#----------------------
# simulation parameters

t = 0.005  # simulation time

#----------------------
# gaussian parameters

sigma = 0.02 # std
x_0 = 0.3   # starting position 
k_0 = 500   # initial momentum

# boundary conditions

boundary = 'periodic'   # 'periodic'

#-----------------------
# potential parameters

potential = 'barrier'   # 'flat', 'barrier', 'harmonic', 'delta'

# potential barrier / well

b = 0.01 # half width 
h = 2.5e5  # height: h < 0 = potential well

# harmonic potential 

a = 1e5   # aperture

# delta potential

alpha = 2e5    # height

# ----------------------
# plotting parameters

# file format
file_format = ''  # 'gif', 'mp4', ''

# animation speed multiplier

play_speed = 100 # video duration (seconds) = t * 2e5 / speed