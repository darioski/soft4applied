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
t = float(config.get('settings', 't'))
dt = float(config.get('settings', 'dt'))
dx = float(config.get('settings', 'dx'))
potential = config.get('settings', 'potential')
x_0 = float(config.get('settings', 'x_0'))
sigma = float(config.get('settings', 'sigma'))
k_0 = float(config.get('settings', 'k_0'))
n = int(1 / dx)
m = abs(int(t / dt))

# control ---> raise exception if:
# there are no timesteps to be performed
if m == 0:
	raise ValueError('Input parameter \'t\' is too small. Choose \'t\' bigger than {:1.1e}'.format(dt))
# starting position is out of range
if not 0. < x_0 < 1.:
	raise ValueError('Input parameter \'x_0\' is out of range (0, 1).')
# rms of the gaussian is smaller than 3 * dx
if sigma < 3 * dx:
	raise ValueError('Chosen \'sigma\' is too small.')
# the gaussian is too close to the edge (tolerance = 6 * sigma)
if x_0 < 6 * sigma or 1 - x_0 < 6 * sigma:
	raise ValueError('Wave-packet is too close to the edge.\n Choose a different \'x_0\' or a smaller \'sigma\'.')
# initial momentum is out of range (k_max = pi * n, tolerance = 3 / sigma)
if np.pi * n - abs(k_0) < 3 / sigma:
	raise ValueError('Chosen \'k_0\' is too large.')

# read filepaths and create parent directories
filepath_1 = config.get('paths', 'pot')
filepath_2 = config.get('paths', 'psi_2')
filepath_3 = config.get('paths', 'phi_2')
filepath_4 = config.get('paths', 'statistics')
Path(filepath_1).parent.mkdir(parents=True, exist_ok=True)
Path(filepath_2).parent.mkdir(parents=True, exist_ok=True)
Path(filepath_3).parent.mkdir(parents=True, exist_ok=True)
Path(filepath_4).parent.mkdir(parents=True, exist_ok=True)

# create real and reciprocal space
x = np.linspace(0., 1., n, endpoint=False)
k = 2 * np.pi * np.fft.fftfreq(n, d=dx)

# create potential profile
pot = np.zeros(n)
if potential == 'barrier':
	b = float(config.get('settings', 'b'))
	h = float(config.get('settings', 'h'))
	pot = wp.barrier_potential(x, b, h)
if potential == 'harmonic':
	a = float(config.get('settings', 'a'))
	pot = wp.harmonic_potential(x, a)
if potential == 'delta':
	alpha = float(config.get('settings', 'alpha'))
	pot = wp.barrier_potential(x, dx, alpha)





# set initial state
psi = np.empty((n, m+1), dtype=complex)
phi = np.empty((n, m+1), dtype=complex)
psi[:, 0] = wp.initial_state(x, x_0, sigma, k_0)
phi[:, 0] = np.fft.fft(psi[:, 0])

# apply algorithm
print("System evolving...  this may take a while...")
for j in range(m):
	psi[:, j+1] = wp.timestep(psi[:, j], pot, k, dt) 
	phi[:, j+1] = np.fft.fft(psi[:, j+1])

# squared module (probability density)
psi_2 = np.abs(psi) ** 2
phi_2 = 1 / (2 * np.pi * n ** 2) * np.abs(phi) ** 2

# statistics
p_left, x_mean, x_rms = wp.x_stats(psi_2, x)
pk_left, k_mean, k_rms = wp.k_stats(phi_2, k)

end_run = time.time()

print("Almost done! Saving data... please wait")

# save in binary files
with open(filepath_1, 'wb') as f:
	np.save(f, pot)
with open(filepath_2, 'wb') as f:
	np.save(f, psi_2)
with open(filepath_3, 'wb') as f:
	np.save(f, phi_2)

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
