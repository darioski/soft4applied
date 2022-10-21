import numpy as np


def check_time_length(t, dt):
    if t < dt:
        raise ValueError("Input parameter \'t\' is too small. Choose \'t\' bigger than {:1.1e}".format(dt))


def is_in_range(x_0):
    if not 0. < x_0 < 1.:
        raise ValueError("Input parameter \'x_0\' is out of range (0, 1).")


def is_wide_enough(sigma, dx):
    if sigma < 3 * dx:
        raise ValueError("Chosen \'sigma\' is too small.")


def is_centered(x_0, sigma):
    # check if gaussian function is not on the edge
    if x_0 < 6 * sigma or 1 - x_0 < 6 * sigma:
        raise ValueError("Wave-packet is too close to the edge.\n\
            Choose a different \'x_0\' or a smaller \'sigma\'.")


def check_initial_momentum(n, sigma, k_0):
    k_max = np.pi * n
    width = 1 / sigma
    if k_max - np.abs(k_0) < 6 * width:
        raise ValueError('Chosen \'k_0\' is too large.\n\
            Choose a value between -{:2.2f} and +{:2.2f}'.format(k_max - 6 * width, k_max - 6 * width) + \
            ', or try a larger \'sigma\'')


def initial_state(x, x_0, sigma, k_0):
    norm = 1. / (2 * np.pi * sigma ** 2) ** 0.25 
    psi = norm * np.exp(1j * k_0 * x - ((x - x_0) / (2 * sigma)) ** 2)
    return psi
    

def potential_operator(psi, pot, dt):
    # half potential energy operator
    return psi * np.exp(-0.5j * dt * pot)


def kinetic_operator(phi, k, dt):
    # kinetic energy operator
    return phi * np.exp(-0.5j * dt * k ** 2)


def timestep(psi, pot, k, dt):
    # Trotter-Suzuki formula + FFT
    psi = potential_operator(psi, pot, dt)  # apply operator V/2
    phi = np.fft.fft(psi)                   # fft to reciprocal space
    phi = kinetic_operator(phi, k, dt)      # apply operator T
    psi = np.fft.ifft(phi)                  # inverse fft to real space
    psi = potential_operator(psi, pot, dt)  # apply operator V/2  
    return psi


def check_boundary(s, l):
    # check if input string in boundary is a valid one
    if s not in l:
        raise ValueError(f"Boundary \'" + s + "\' is not defined. Valid input strings are:\n \
            \'" + "\', \'".join(l) + "\'")


def check_potential(s, l):
    # check if input string in potential is a valid one
    if s not in l:
        raise ValueError("Potential \'" + s + "\' is not defined. Valid input strings are:\n \
            \'" + "\', \'".join(l) + "\'")


def barrier_potential(x, b, h):
    # centered barrier potential
    # barrier if h > 0
    # well if h < 0
    n = len(x)
    pot = np.zeros(n)
    for i, pos in enumerate(x):
        if x[n//2] - b < pos < x[n//2] + b:
            pot[i] = h
    return pot


def harmonic_potential(x, a):
    # centered harmonic potential
    n = len(x)
    return a * (x - x[n//2]) ** 2


def rms_x(x, psi_2, x_m):
    # rms(x)**2 = <x**2> - <x>**2
    x2_m = np.mean(x ** 2 * psi_2)
    return np.sqrt(x2_m - x_m ** 2)

def rms_k(k, phi_2, k_m):
    k2_m = np.sum(k ** 2 * phi_2) * 2 * np.pi
    return np.sqrt(k2_m - k_m ** 2)
