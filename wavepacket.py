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
    

def potential_operator(wavefunction, potential, dt):
    '''
    Apply the potential operator for half timestep dt / 2.

    Parameters
    ----------
    wavefunction : 1d array, complex-valued wavefunction
    potential : 1d array, potential profile
    dt : real, timestep length

    Returns
    -------
    1d array, partially evolved wavefunction after dt / 2.
    '''
    return wavefunction * np.exp(-0.5j * dt * potential)


def kinetic_operator(wavefunction_transform, k, dt):
    '''
    Apply the kinetic operator for one timestep dt.

    Parameters
    ----------
    wavefunction_transform : 1d array, fourier transform of the wavefunction
    k : 1d array, the reciprocal space (momentum space)
    dt : float, timestep length

    Returns
    -------
    1d array, partially evolved wavefunction after dt
    '''
    return wavefunction_transform * np.exp(-0.5j * dt * k ** 2)


def timestep(wavefunction, potential, k, dt):
    '''
    Apply the evolution operator for one timestep dt using 
    Trotter-Suzuki decomposition at 2nd order.

    Parameters
    ----------
    wavefunction : 1d array, complex-valued wavefunction
    potential : 1d array, potential profile
    k : 1d array, the reciprocal space (momentum space)
    dt : float, timestep length

    Returns
    -------
    wavefunction : 1d array, evolved wavefunction after dt

    '''
    # apply potential operator for dt / 2
    wavefunction = potential_operator(wavefunction, potential, dt) 
    # apply fft
    wavefunction_transform = np.fft.fft(wavefunction)
    # apply kinetic operator for dt
    wavefunction_transform = kinetic_operator(wavefunction_transform, k, dt)
    # apply inverse fft
    wavefunction = np.fft.ifft(wavefunction_transform)
    # apply potential operator for dt / 2
    wavefunction = potential_operator(wavefunction, potential, dt) 
    return wavefunction


def barrier_potential(x, half_width, height):
    '''
    Return a square potential at the center of the x space.

    Parameters
    ----------
    x : 1d array, the real space
    half_width : float, half width of the potential barrier
    height : float, energy of the potential barrier.
             if h < 0 ---> potential well

    Returns
    -------
    potential : 1d array, the potential profile
    '''
    n = len(x)
    # return height when the condition is met, 0 elsewhere
    potential = np.where(np.abs(x - x[n//2]) < half_width, height, 0)
    return potential


def harmonic_potential(x, a):
    '''
    Return a centered harmonic potential.

    Parameters
    ----------
    x : 1d array, the real space
    a : float, aperture of the parabola 

    Returns
    -------
    1d array, the harmonic potential profile
    '''
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
