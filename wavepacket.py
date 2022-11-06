import numpy as np


def gaussian_initial_state(x, start_position, sigma, start_momentum):
    '''
    Create a gaussian wavefunction as initial state.

    Parameters
    ----------
    x : 1d array, the real space
    start_position : float, centre of the gaussian
    sigma : float, standard deviation of the gaussian
    start_momentum : float, starting average momentum

    Returns
    -------
    wavefunction : 1d array, gaussian wavefunction
    '''
    # normalization factor
    norm_factor = 1. / (2 * np.pi * sigma ** 2) ** 0.25 
    # gaussian wavefunction
    wavefunction = norm_factor * np.exp(1j * start_momentum * x - ((x - start_position) / (2 * sigma)) ** 2)
    return wavefunction
    

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


def x_stats(psi_2, x):
    '''
    Compute statistical quantities in the real space.

    Parameters
    ----------
    psi_2 : 2d array, probability density function.
    x : 1d array, the real space.

    Returns
    -------
    p_left : 1d array, probability for x < 0.5 at each timestep.
    x_mean : 1d array, average position at each timestep.
    x_rms : 1d array, standard deviation from x_mean at each timestep.
    '''

    n = len(x)
    m = psi_2.shape[1] - 1
    p_left = np.empty(m+1)
    x_mean = np.empty(m+1)
    x_rms = np.empty(m+1)

    for j in range(m+1):
        p_left[j] = 0.5 * np.mean(psi_2[:n//2, j])
        x_mean[j] = np.mean(x * psi_2[:, j])
        # rms(x) ** 2 = < x ** 2 > - < x > ** 2
        x2_m = np.mean(x ** 2 * psi_2[:, j])
        x_rms[j] = np.sqrt(x2_m - x_mean[j] ** 2)
    return p_left, x_mean, x_rms

    

def k_stats(phi_2, k):
    '''
    Compute statistical quantities in the reciprocal space.

    Parameters
    ----------
    phi_2 : 2d array, probability density function in reciprocal space.
    k : 1d array, the reciprocal space.

    Returns
    -------
    pk_left : 1d array, probability for k < 0 at each timestep.
    k_mean : 1d array, average momentum at each timestep.
    k_rms : 1d array, standard deviation from k_mean at each timestep.
    '''

    n = len(k)
    m = phi_2.shape[1] - 1
    pk_left = np.empty(m+1)
    k_mean = np.empty(m+1)
    k_rms = np.empty(m+1)  

    for j in range(m+1):
        pk_left[j] = 0.5 * np.sum(phi_2[n//2:, j]) * 2 * np.pi
        k_mean[j] = np.sum(k * phi_2[:, j]) * 2 * np.pi
        k2_m = np.sum(k ** 2 * phi_2[:, j]) * 2 * np.pi
        k_rms[j] = np.sqrt(k2_m - k_mean[j] ** 2)
    return pk_left, k_mean, k_rms
