# Spreading of a wave-packet

## Input parameters _(params.py)_


```python
#----------------------
# simulation parameters

t = 0.01  # simulation time

#----------------------
# gaussian parameters

sigma = 0.02 # std
x_0 = 0.5    # starting position 
k_0 = 1000   # initial momentum

# boundary conditions

boundary = 'periodic'   # 'periodic'

#-----------------------
# potential parameters

potential = 'flat'   # 'flat', 'barrier', 'harmonic', 'delta'

# ----------------------
# plotting parameters

# file format
file_format = 'gif'  # 'gif', 'mp4'

# animation speed multiplier

play_speed = 50 # video duration (seconds) = t * 2e5 / speed
```

## Animation

<img src="../gifs/spreading.gif" width=600 height=400 />


## Data analysis


```python
import pandas as pd

stats = pd.read_csv('statistics.csv')

print('Start:')
print(stats.iloc[[0]])
print('End:')
print(stats.iloc[[-1]])
```

    Start:
       time   p_left  x_mean  x_rms       pk_left  k_mean  k_rms
    0   0.0  0.49026     0.5   0.02  1.213437e-32   500.0   25.0
    End:
            time    p_left    x_mean     x_rms       pk_left  k_mean  k_rms
    50000  0.005  0.501532  0.498461  0.406225  9.958769e-28   500.0   25.0

