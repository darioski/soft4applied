import numpy as np
from pytest import param
import myfunctions as mf
import params
import time
from scipy.optimize import curve_fit


start_time = time.time()

# set parameters
dt = 1e-7
mf.check_time_length(params.t, dt)

m = int(1e7 * params.t)
n = 1024

# reciprocal space
k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

# boundary conditions
boundary_list = ['periodic']
mf.check_boundary(params.boundary, boundary_list)

if params.boundary == 'periodic':
    x = np.linspace(0., 1., n, endpoint=False)
    dx = x[1]

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


# check start position
mf.is_in_range(params.x_0)
mf.is_wide_enough(params.sigma, dx)
mf.is_centered(params.x_0, params.sigma)

# define wave functions
psi = np.zeros((n, m+1), dtype=complex)
phi = np.zeros((n, m+1), dtype=complex)

norm = 1. / (2 * np.pi * params.sigma ** 2) ** 0.25    # normalization
psi[:, 0] = norm * np.exp(1j * params.k_0 * x - ((x - params.x_0) / (2 * params.sigma)) ** 2)
phi[:, 0] = np.fft.fft(psi[:, 0])

print("System evolving...  this may take a while...")

# evolve
for j in range(m):
    psi[:, j+1] = mf.timestep(psi[:, j], pot, k, dt) 
    phi[:, j+1] = np.fft.fft(psi[:, j+1])


# compute squared module
psi_2 = np.abs(psi) ** 2
phi_2 = 1 / (2 * np.pi * n ** 2) * np.abs(phi) ** 2

end_run = time.time()

print("Almost done! Saving data... please wait")

# save in output files
with open('x.npy', 'wb') as f:
    np.save(f, x)

with open('pot.npy', 'wb') as f:
    np.save(f, pot)

with open('psi_2.npy', 'wb') as f:
    np.save(f, psi_2)

end_load = time.time()

# analysis file
if params.potential == 'flat':

    x_f = np.average(x, weights=psi_2[:, -1])       # final position
    popt, pcov = curve_fit(mf.gaussian, x, psi_2[:, -1], p0=[x_f, params.sigma])
    E_f = 0.5 * np.average(k, weights=phi_2[:, -1]) ** 2

    with open('analysis.out', 'w') as f:
        f.write(str(np.trapz(psi_2[:, 0], x)) + '\n')
        f.write(str(np.trapz(psi_2[:, -1], x)) + '\n')
        f.write('Potential = flat\n')
        f.write('Start:\n')
        f.write('x_mean = {:1.3f}\n'.format(params.x_0))
        f.write('std = {:1.3f}\n'.format(params.sigma))
        f.write('E_mean = {:2.3e}\n'.format(0.5 * params.k_0 ** 2))
        f.write('End:\n')
        f.write('x_mean = {:1.3f}\n'.format(x_f))
        f.write('std = {:1.3f}\n'.format(popt[1]))
        f.write('E_mean = {:2.3e}\n'.format(E_f))


print("Runtime = {:5.3f} s".format(end_run - start_time))
print("Data saving time = {:5.3f} s".format(end_load - end_run))
print("Total runtime = {:5.3f} s".format(end_load - start_time))