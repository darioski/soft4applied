import numpy as np
import pickle
import pytest
from myfunctions import timestep, potential_barrier, harmonic_potential
import params


# set parameters
n = 1024
dt = 1e-7
m = int(1e7 * params.t)

# reciprocal space
k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

# boundary conditions
if params.boundary == 'periodic':
    x = np.linspace(0., 1., n, endpoint=False)

# choose potential
if params.potential == 'flat':
    pot = np.zeros(n)

if params.potential == 'barrier':
    pot = potential_barrier(x, params.b, params.h)

if params.potential == 'harmonic':
    pot = harmonic_potential(x, params.a)


# define wave functions
psi = np.zeros((n, m+1), dtype=complex)
phi = np.zeros((n, m+1), dtype=complex)

norm = 1. / (2 * np.pi * params.sigma ** 2) ** 0.25    # normalization
psi[:, 0] = norm * np.exp(1j * params.k_0 * x - ((x - params.x_0) / (2 * params.sigma)) ** 2)
phi[:, 0] = np.fft.fft(psi[:, 0])

# evolve
for j in range(m):
    psi[:, j+1], phi[:, j+1] = timestep(psi[:, j], pot, k, dt) 


# compute squared module
psi_2 = np.zeros((n, m+1))
phi_2 = np.zeros((n, m+1))

for j in range(m+1):
    psi_2[:, j] = np.abs(psi[:, j]) ** 2
    phi_2[:, j] = 1 / (2 * np.pi * n ** 2) * np.abs(phi[:, j]) ** 2

# save in output file
data = {'x':x, 'k':k, 'pot':pot, 'psi':psi, 'phi':phi, 'psi_2':psi_2, 'phi_2':phi_2}
with open('data.pickle', 'wb') as datafile:
    pickle.dump(data, datafile, pickle.HIGHEST_PROTOCOL)


# probability analysis
print(np.trapz(psi_2[:, 0], x))
print(np.trapz(psi_2[:, -1], x))
print(np.trapz(psi_2[:n//2, -1], x[:n//2]))     # integral on left side
print(np.trapz(psi_2[n//2:, -1], x[n//2:]))     # integral on right side


# --------- tests ----------

def test_length():
    assert len(psi) == len(x)

def test_psi_normalization():
    i = np.trapz(psi_2[:, 0], x)
    assert np.isclose(i, 1.)

def test_phi_normalization():
    i = np.trapz(phi_2[:, 0], k)
    assert np.isclose(i, 1.)

def test_norm_conserved():
    i_start = np.trapz(psi_2[:, 0], x)
    i_end =  np.trapz(psi_2[:, -1], x)
    assert np.isclose(i_start, i_end)