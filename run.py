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

# wave packet
psi, phi = initial_conditions(x)

# compute squared module
psi_2 = np.abs(psi) ** 2
f = 1 / 0.4096  # correction factor 
phi_2 = 1 / (2 * np.pi * n) * np.abs(phi) ** 2 * f

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