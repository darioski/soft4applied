from mimetypes import init
import numpy as np
import pytest


def potential_operator(psi, pot, dt):
    return psi * np.exp(-0.5j * dt * pot)


def kinetic_operator(phi, k, dt):
    return phi * np.exp(-0.5j * dt * k ** 2)


def time_step(psi, pot, k, dt, n):
    
    psi = potential_operator(psi, pot, dt)  # apply operator V/2
    phi = np.fft.fft(psi, n)   # fft to reciprocal space
    phi = kinetic_operator(phi, k, dt)  # apply operator T
    psi = np.fft.ifft(phi, n)  # inverse fft to real space
    psi = potential_operator(psi, pot, dt)  # apply operator V/2
    phi = np.fft.fft(psi, n)   # fft
    
    return psi, phi


# ----------- tests ------------

