import numpy as np


def gaussian(x, x_0, sigma):
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-0.5 * (x - x_0) ** 2 / sigma ** 2)


def check_simulation_time(t, dt):
    if t < dt:
        raise ValueError("Input parameter \"t\" is too small. Choose \"t\" bigger than {:1.1e}".format(dt))


def check_start_condition(x_0, dx, sigma):
    if not 0. < x_0 < 1.:
        raise ValueError("Input parameter \"x_0\" is out of range (0, 1).")
    if sigma < 3 * dx:
        raise ValueError("Chosen \"sigma\" is too small.")
    if x_0 < 6 * sigma or 1 - x_0 < 6 * sigma:
        raise ValueError("Wave-packet is too close to the edge. Choose a different \"x_0\" or a smaller \"sigma\".")
    

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

