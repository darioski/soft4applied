# Wave-packet in a potential well

## Input parameters _(params.py)_

```python
#----------------------
# simulation parameters

t = 0.01  # simulation time

#----------------------
# gaussian parameters

sigma = 0.02 # std
x_0 = 0.5    # starting position 
k_0 = 0   # initial momentum

# boundary conditions

boundary = 'periodic'   # 'periodic'

#-----------------------
# potential parameters

potential = 'barrier'   # 'flat', 'barrier', 'harmonic', 'delta'

# potential barrier / well

b = 0.15 # half width 
h = -1e7  # height: h < 0 = potential well

# ----------------------
# plotting parameters

# file format
file_format = 'gif'  # 'gif', 'mp4'

# animation speed multiplier

play_speed = 100 # video duration (seconds) = t * 2e5 / speed

```

## Animation

<img src="../gifs/well.gif" width=600 height=400 />
