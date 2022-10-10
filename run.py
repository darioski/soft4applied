import numpy as np
import myfunctions as mf
import params
import time
from scipy.optimize import curve_fit


start_time = time.time()

# set parameters
dt = 1e-7
mf.check_simulation_time(params.t, dt)
m = int(1e7 * params.t)
n = 1024

# reciprocal space
k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

# boundary conditions
if params.boundary == 'periodic':
    x = np.linspace(0., 1., n, endpoint=False)
    dx = x[1]

# choose potential
if params.potential == 'flat':
    pot = np.zeros(n)

if params.potential == 'barrier':
    pot = mf.potential_barrier(x, params.b, params.h)

if params.potential == 'harmonic':
    pot = mf.harmonic_potential(x, params.a)


mf.check_start_condition(params.x_0, dx, params.sigma)

# define wave functions
psi = np.zeros((n, m+1), dtype=complex)
phi = np.zeros((n, m+1), dtype=complex)

norm = 1. / (2 * np.pi * params.sigma ** 2) ** 0.25    # normalization
psi[:, 0] = norm * np.exp(1j * params.k_0 * x - ((x - params.x_0) / (2 * params.sigma)) ** 2)
phi[:, 0] = np.fft.fft(psi[:, 0])



print("System evolving...  this may take a while...")

# evolve
for j in range(m):
    psi[:, j+1], phi[:, j+1] = mf.timestep(psi[:, j], pot, k, dt) 


# compute squared module
psi_2 = np.zeros((n, m+1))
phi_2 = np.zeros((n, m+1))

for j in range(m+1):
    psi_2[:, j] = np.abs(psi[:, j]) ** 2
    phi_2[:, j] = 1 / (2 * np.pi * n ** 2) * np.abs(phi[:, j]) ** 2

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