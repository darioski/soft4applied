import numpy as np


# real space sampling
def xspace(le, n):
    return np.linspace(0, le, n, endpoint=False)


# reciprocal space sampling
#def kspace(le, n):
#    return 2 * np.pi * np.fft.fftfreq(n, d=le/n)


# set wavepacket initial condition
def initial_conditions(sigma, kappa, psi, x, x_0):
    a = 1 / (2 * np.pi * sigma ** 2) ** 0.25    # normalization
    psi[:, 0] = a * np.exp(1j * kappa * x - ((x - x_0) / (2 * sigma )) ** 2)


