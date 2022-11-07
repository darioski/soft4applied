import numpy as np
import wavepacket as wp
import time
import pandas as pd
import configparser
from pathlib import Path
import sys

start_time = time.time()

# read parameters
config = configparser.ConfigParser()
config_file = sys.argv[1]
config.read(config_file)
t = float(config.get('Simulation settings', 't'))
dt = float(config.get('Simulation settings', 'dt'))
dx = float(config.get('Simulation settings', 'dx'))
potential_type = config.get('Potential profile type', 'potential_type')
start_position = float(config.get('Initial state settings', 'start_position'))
sigma = float(config.get('Initial state settings', 'sigma'))
start_momentum = float(config.get('Initial state settings', 'start_momentum'))
n = int(1 / dx)
m = abs(int(t / dt))

# control ---> raise exception if:
# there are no timesteps to be performed
if m == 0:
	raise ValueError('Input parameter \'t\' is too small. Choose \'t\' bigger than {:1.1e}'.format(dt))
# starting position is out of range
if not 0. < start_position < 1.:
	raise ValueError('Input parameter \'x_0\' is out of range (0, 1).')
# rms of the gaussian is smaller than 3 * dx
if sigma < 3 * dx:
	raise ValueError('Chosen \'sigma\' is too small.')
# the gaussian is too close to the edge (tolerance = 6 * sigma)
if start_position < 6 * sigma or 1 - start_position < 6 * sigma:
	raise ValueError('Wave-packet is too close to the edge.\n Choose a different \'x_0\' or a smaller \'sigma\'.')
# initial momentum is out of range (k_max = pi * n, tolerance = 3 / sigma)
if np.pi * n - abs(start_momentum) < 3 / sigma:
	raise ValueError('Chosen \'k_0\' is too large.')

# read filepaths and create parent directories
filepath_1 = config.get('Paths to files', 'potential')
filepath_2 = config.get('Paths to files', 'probability')
filepath_3 = config.get('Paths to files', 'transform_probability')
filepath_4 = config.get('Paths to files', 'statistics')
Path(filepath_1).parent.mkdir(parents=True, exist_ok=True)
Path(filepath_2).parent.mkdir(parents=True, exist_ok=True)
Path(filepath_3).parent.mkdir(parents=True, exist_ok=True)
Path(filepath_4).parent.mkdir(parents=True, exist_ok=True)

# create real and reciprocal space
x = np.linspace(0., 1., n, endpoint=False)
k = 2 * np.pi * np.fft.fftfreq(n, d=dx)

# create potential profile
potential = np.zeros(n)
if potential_type == 'barrier':
	half_width = float(config.get('Barrier potential', 'half_width'))
	height = float(config.get('Barrier potential', 'height'))
	potential = wp.barrier_potential(x, half_width, height)
if potential_type == 'harmonic':
	aperture = float(config.get('Harmonic potential', 'aperture'))
	potential = wp.harmonic_potential(x, aperture)
if potential_type == 'delta':
	alpha = float(config.get('Delta potential', 'alpha'))
	potential = wp.barrier_potential(x, dx, alpha)


# set initial state
wavefunction = np.empty((n, m+1), dtype=complex)
wavefunction_transform = np.empty((n, m+1), dtype=complex)
wavefunction[:, 0] = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)
wavefunction_transform[:, 0] = np.fft.fft(wavefunction[:, 0])

# apply algorithm
print("System evolving...  this may take a while...")
for j in range(m):
	wavefunction[:, j+1] = wp.timestep(wavefunction[:, j], potential, k, dt) 
	wavefunction_transform[:, j+1] = np.fft.fft(wavefunction[:, j+1])

# squared module (probability density)
probability = np.abs(wavefunction) ** 2
transform_probability = 1 / (2 * np.pi * n ** 2) * np.abs(wavefunction_transform) ** 2

# statistics
p_left, x_mean, x_rms = wp.x_stats(probability, x)
pk_left, k_mean, k_rms = wp.k_stats(transform_probability, k)

end_run = time.time()

print("Almost done! Saving data... please wait")

# save in binary files
with open(filepath_1, 'wb') as f:
	np.save(f, potential)
with open(filepath_2, 'wb') as f:
	np.save(f, probability)
with open(filepath_3, 'wb') as f:
	np.save(f, transform_probability)

# write statistics as csv
d = {'time':dt*np.arange(m+1), 'p_left':p_left, 'x_mean':x_mean, 'x_rms':x_rms, 
     'pk_left':pk_left, 'k_mean':k_mean, 'k_rms':k_rms}
stats = pd.DataFrame(data=d)
stats.to_csv(filepath_4, index=False)

end_save = time.time()

# log
print(f'Wavepacket array dimension = ({n} x {m})')
print("Algorithm runtime = {:5.3f} s".format(end_run - start_time))
print("Total runtime = {:5.3f} s".format(end_save - start_time))
