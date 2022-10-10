import numpy as np
import pytest


def gaussian(x, a, b, sigma):
    return a * np.exp(- 0.5 * (x - b) ** 2 / sigma ** 2)


def potential_operator(psi, pot, dt):
    return psi * np.exp(-0.5j * dt * pot)


def kinetic_operator(phi, k, dt):
    return phi * np.exp(-0.5j * dt * k ** 2)


def timestep(psi, pot, k, dt):
    
    psi = potential_operator(psi, pot, dt)  # apply operator V/2
    phi = np.fft.fft(psi)   # fft to reciprocal space
    phi = kinetic_operator(phi, k, dt)  # apply operator T
    psi = np.fft.ifft(phi)  # inverse fft to real space
    psi = potential_operator(psi, pot, dt)  # apply operator V/2
    phi = np.fft.fft(psi)   # fft
    
    return psi, phi


def potential_barrier(x, b, h):

    n = len(x)
    pot = np.zeros(n)
    for i, pos in enumerate(x):
        if 0.5 - b < pos < 0.5 + b:
            pot[i] = h
    return pot


def harmonic_potential(x, a):
    return a * (x - 0.5) ** 2


# ----------- tests ------------

