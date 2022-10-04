from mimetypes import init
import numpy as np
from myfunctions import *
import pytest

# read parameters from file
le, n = np.loadtxt('input', usecols=2, unpack=True)
n = int(n)

# x-space
x = np.linspace(0, le, n, endpoint=False)

# reciprocal space
k = 2 * np.pi * np.fft.fftfreq(n, d=le/n)

# read parameters for wave packet
sigma = 1
a = 1 / (2 * np.pi * sigma ** 2) ** 0.25    # normalization
x_0 = 10    # initial position
k_0 = 10    # initial momentum

# define wave functions
psi = a * np.exp(1j * k_0 * x - ((x - x_0) / (2 * sigma)) ** 2)
phi = np.fft.fft(psi)


# compute squared module
psi_2 = np.abs(psi) ** 2
phi_2 = le ** 2 / (2 * np.pi * n ** 2) * np.abs(phi) ** 2

# save in output files
with open('psi_2.npy', 'wb') as f:
    np.save(f, psi_2)

with open('phi_2.npy', 'wb') as f:
    np.save(f, phi_2)

# --------- tests ----------

def test_length():
    assert len(psi) == n

def test_type():
    assert psi.dtype == 'complex128'

def test_psi_normalization():
    i = np.trapz(psi_2, x)
    assert np.isclose(i, 1.)

def test_phi_normalization():
    i = np.trapz(phi_2, k)
    assert np.isclose(i, 1.)