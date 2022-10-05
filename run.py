import numpy as np
import pickle
import pytest
from myfunctions import *



# read parameters from file
n, m = np.loadtxt('input', usecols=2, unpack=True)
n = int(n)
m = int(m)
dt = 1e-3 / m

# x-space
x = np.linspace(0., 1., n, endpoint=False)

# potential
pot = np.zeros(n)

# reciprocal space
k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

# read parameters for wave packet
sigma = 0.01
a = 1. / (2 * np.pi * sigma ** 2) ** 0.25    # normalization
x_0 = 0.1    # initial position
k_0 = 1000  # initial momentum

# define wave functions
psi = np.zeros((n, m+1), dtype=complex)
phi = np.zeros((n, m+1), dtype=complex)

psi[:, 0] = a * np.exp(1j * k_0 * x - ((x - x_0) / (2 * sigma)) ** 2)
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
data = {'psi':psi, 'phi':phi, 'psi_2':psi_2, 'phi_2':phi_2}
with open('data.pickle', 'wb') as datafile:
    pickle.dump(data, datafile, pickle.HIGHEST_PROTOCOL)


# --------- tests ----------

def test_length():
    assert len(psi) == n

def test_psi_normalization():
    i = np.trapz(psi_2[:, 0], x)
    assert np.isclose(i, 1.)

def test_phi_normalization():
    i = np.trapz(phi_2[:, 0], k)
    assert np.isclose(i, 1.)

def test_norm_conserved():
    i_start = np.trapz(psi_2[:, 0], x)
    i_end =  np.trapz(psi_2[:, -1], x)
    assert np.isclose(i_start, i_end, atol=1e-5)