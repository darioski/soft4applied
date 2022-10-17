import numpy as np
import myfunctions as mf
import params
import time


start_time = time.time()

# set parameters
dt = 1e-7
mf.check_time_length(params.t, dt)

m = int(1e7 * params.t)
n = 1024

# 
x = np.linspace(0., 1., n, endpoint=False)
dx = x[1] - x[0]

# reciprocal space
k = 2 * np.pi * np.fft.fftfreq(n, d=dx)

# check potential
potential_list = ['flat', 'barrier', 'harmonic', 'delta']
mf.check_potential(params.potential, potential_list)

# choose potential
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

# check start position
mf.is_in_range(params.x_0)
mf.is_wide_enough(params.sigma, dx)
mf.is_centered(params.x_0, params.sigma)

mf.check_initial_momentum(n, params.sigma, params.k_0)

# define wave functions
psi = np.empty((n, m+1), dtype=complex)
phi = np.empty((n, m+1), dtype=complex)

# set initial state
psi[:, 0] = mf.initial_state(x, params.x_0, params.sigma, params.k_0)
phi[:, 0] = np.fft.fft(psi[:, 0])

# evolve
print("System evolving...  this may take a while...")

for j in range(m):
	psi[:, j+1] = mf.timestep(psi[:, j], pot, k, dt) 
	phi[:, j+1] = np.fft.fft(psi[:, j+1])

# compute squared module
psi_2 = np.abs(psi) ** 2
phi_2 = 1 / (2 * np.pi * n ** 2) * np.abs(phi) ** 2

# compute statistics
p_left = np.empty(m+1)
x_mean = np.empty(m+1)
x_rms = np.empty(m+1)
for j in range(m+1):
	p_left[j] = np.mean(psi_2[:n//2, j])
	x_mean[j] = np.mean(x * psi_2[:, j])
	x_rms[j] = mf.rms_x(x, psi_2[:, j], x_mean[j])

end_run = time.time()

print("Almost done! Saving data... please wait")

# save in output files
with open('pot.npy', 'wb') as f:
	np.save(f, pot)
with open('psi_2.npy', 'wb') as f:
	np.save(f, psi_2)
with open('p_left.npy', 'wb') as f:
	np.save(f, p_left)
with open('x_mean.npy', 'wb') as f:
	np.save(f, x_mean)
with open('x_rms.npy', 'wb') as f:
	np.save(f, x_rms)

with open('phi_2.npy', 'wb') as f:
	np.save(f, phi_2)


end_save = time.time()

print(f'Wavepacket array dimension = ({n} x {m})')
print("Algorithm runtime = {:5.3f} s".format(end_run - start_time))
print("Total runtime = {:5.3f} s".format(end_save - start_time))
