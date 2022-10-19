import numpy as np
import myfunctions as mf
import params
import time
import pandas as pd


start_time = time.time()

# set parameters
dt = 1e-7
n = 1024
m = int(params.t / dt)
dx = 1 / n

mf.check_time_length(params.t, dt)

# real and momenta space
x = np.linspace(0., 1., n, endpoint=False)
k = 2 * np.pi * np.fft.fftfreq(n, d=dx)

# choose potential
potential_list = ['flat', 'barrier', 'harmonic', 'delta']
mf.check_potential(params.potential, potential_list)
if params.potential == 'flat':
	pot = np.zeros(n)
if params.potential == 'barrier':
	pot = mf.barrier_potential(x, params.b, params.h)
if params.potential == 'harmonic':
	pot = mf.harmonic_potential(x, params.a)
if params.potential == 'delta':
	pot = mf.barrier_potential(x, dx, params.alpha)

# boundary conditions
boundary_list = ['periodic']
mf.check_boundary(params.boundary, boundary_list)

# check initial conditions
mf.is_in_range(params.x_0)
mf.is_wide_enough(params.sigma, dx)
mf.is_centered(params.x_0, params.sigma)
mf.check_initial_momentum(n, params.sigma, params.k_0)

# set initial state
psi = np.empty((n, m+1), dtype=complex)
phi = np.empty((n, m+1), dtype=complex)
psi[:, 0] = mf.initial_state(x, params.x_0, params.sigma, params.k_0)
phi[:, 0] = np.fft.fft(psi[:, 0])

# apply algorithm
print("System evolving...  this may take a while...")
for j in range(m):
	psi[:, j+1] = mf.timestep(psi[:, j], pot, k, dt) 
	phi[:, j+1] = np.fft.fft(psi[:, j+1])

# squared module (probability density)
psi_2 = np.abs(psi) ** 2
phi_2 = 1 / (2 * np.pi * n ** 2) * np.abs(phi) ** 2

# statistics
p_left = np.empty(m+1)
x_mean = np.empty(m+1)
x_rms = np.empty(m+1)
pk_left = np.empty(m+1)
k_mean = np.empty(m+1)
k_rms = np.empty(m+1)

for j in range(m+1):
	p_left[j] = 0.5 * np.mean(psi_2[:n//2, j])
	x_mean[j] = np.mean(x * psi_2[:, j])
	x_rms[j] = mf.rms_x(x, psi_2[:, j], x_mean[j])
	pk_left[j] = 0.5 * np.sum(phi_2[n//2:, j]) * 2 * np.pi
	k_mean[j] = np.sum(k * phi_2[:, j]) * 2 * np.pi
	k_rms[j] = mf.rms_k(k, phi_2[:, j], k_mean[j])

end_run = time.time()

print("Almost done! Saving data... please wait")

# save in binary files
with open('pot.npy', 'wb') as f:
	np.save(f, pot)
with open('psi_2.npy', 'wb') as f:
	np.save(f, psi_2)
with open('phi_2.npy', 'wb') as f:
	np.save(f, phi_2)

# write statistics as csv
d = {'time':dt*np.arange(m+1), 'p_left':p_left, 'x_mean':x_mean, 'x_rms':x_rms, 
     'pk_left':pk_left, 'k_mean':k_mean, 'k_rms':k_rms}
stats = pd.DataFrame(data=d)
stats.to_csv('statistics.csv', index=False)

end_save = time.time()

# log
print(f'Wavepacket array dimension = ({n} x {m})')
print("Algorithm runtime = {:5.3f} s".format(end_run - start_time))
print("Total runtime = {:5.3f} s".format(end_save - start_time))
