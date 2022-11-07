import wavepacket as wp
import numpy as np
from scipy.optimize import curve_fit


# gaussian_initial_state

def test_initial_state_normalized():
    '''
    Test if the sum of probabilities of the initial state adds up to 1.

    GIVEN: a gaussian wavefunction 
    WHEN: I compute the integral of its squared module along x
    THEN: the integral must be equal to 1.
    '''
    # set initial state
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    start_position = 0.4
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)

    # probability and integral along x
    probability = np.abs(wavefunction) ** 2
    I = np.trapz(probability, x)

    assert np.isclose(I, 1)


def test_initial_state_center():
    '''
    Test if computed average position coincides with the center of the gaussian.

    GIVEN: a gaussian wavefunction centered in x = start_position
    WHEN: I compute the average of the position
    THEN: the average position must coincide with the center of the gaussian.
    '''
    # set initial state
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    start_position = 0.4
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)

    # probability and average position
    probability = np.abs(wavefunction) ** 2
    x_mean = np.mean(x * probability)
       
    assert np.isclose(start_position, x_mean, atol=1e-3)
    
    

def test_initial_state_rms():
    '''
    Test if computed rms of position coincides with the set std parameter of the gaussian.

    GIVEN: a gaussian wavefunction with standard deviation sigma
    WHEN: I compute the rms of the position
    THEN: the rms must coincide with sigma.  
    '''
    # set initial state
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    start_position = 0.4
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)

    # probability and rms
    probability = np.abs(wavefunction) ** 2
    x_mean = np.mean(x * probability)
    x_rms = np.sqrt(np.mean(x**2 * probability) - x_mean ** 2)

    assert np.isclose(sigma, x_rms, atol=1e-3)


def gaussian(x, norm, center, standard_dev):
    '''
    Return a gaussian function.
    This is a function for testing.
    '''
    return norm * np.exp(- 0.5 * (x - center) ** 2 / standard_dev ** 2)

def test_initial_state_transform_is_gaussian():
    '''
    Test if the fourier transform of the initial state is a gaussian.

    GIVEN: a gaussian wavefunction and its fft
    WHEN: I perform a gaussian fitting of the fft squared module 
    THEN: The fit should return the theoretical values and small standard deviation errors.
    '''
    # set real and reciprocal space
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

    # set initial state
    start_position = 0.4
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)
    
    # fft and gaussian fit
    wavefunction_transform = np.fft.fft(wavefunction)
    transform_probability = 1 / (2 * np.pi * n ** 2) * np.abs(wavefunction_transform) ** 2
    normalization_factor = np.sqrt(2 * sigma ** 2 / np.pi)
    initial_guess = np.array([normalization_factor, start_momentum, 1 / (2 * sigma)])
    popt, pcov = curve_fit(gaussian, k, transform_probability, p0=initial_guess)

    assert np.all(np.isclose(popt, initial_guess))
    assert np.all(np.isclose(np.sqrt(np.diag(pcov)), 0))


def test_initial_state_transform_normalized():
    '''
    Test if the transform of the initial state is normalized.

    GIVEN: a gaussian wavefunction and its fft
    WHEN: I compute the fft squared module and its integral along k.
    THEN: the integral should be equal to 1.
    '''
    # set real and reciprocal space
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

    # set initial state
    start_position = 0.4
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)
    
    # fft and average momentum
    wavefunction_transform = np.fft.fft(wavefunction)
    transform_probability = 1 / (2 * np.pi * n ** 2) * np.abs(wavefunction_transform) ** 2
    I = np.trapz(transform_probability, k)

    assert np.isclose(I, 1)


def test_initial_state_transform_center():
    '''
    Test if the center of the transformed initial state coincides with the set initial momentum.

    GIVEN: a gaussian wavefunction with initial momentum k = start_momentum
    WHEN: I compute the fft and its average momentum
    THEN: the average momentum should coincide with start_momentum
    '''
    # set real and reciprocal space
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

    # set initial state
    start_position = 0.4
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)
    
    # fft and average momentum
    wavefunction_transform = np.fft.fft(wavefunction)
    transform_probability = 1 / (2 * np.pi * n ** 2) * np.abs(wavefunction_transform) ** 2
    k_mean = np.sum(k * transform_probability) * 2 * np.pi

    assert np.isclose(start_momentum, k_mean, atol=1e-3)


def test_initial_state_transform_rms():
    '''
    Test if rms of the transformed initial state coincides with its theoretical value.

    GIVEN: a gaussian wavefunction with standard deviation sigma and its fft
    WHEN: I compute the fft squared module and its rms
    THEN: the rms should be equal to 1 / (2 * sigma).
    '''
    # set real and reciprocal space
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

    # set initial state
    start_position = 0.4
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)
    
    # fft and rms
    wavefunction_transform = np.fft.fft(wavefunction)
    transform_probability = 1 / (2 * np.pi * n ** 2) * np.abs(wavefunction_transform) ** 2
    k_mean = 2 * np.pi * np.sum(k * transform_probability)
    k2_mean = 2 * np.pi * np.sum(k ** 2 * transform_probability)
    k_rms = np.sqrt(k2_mean - k_mean ** 2)

    assert np.isclose(1 / (2 * sigma), k_rms, atol=1e-3)


# potential_operator

def test_potential_operator_on_probability_distribution():
    '''
    Test if the potenial operator leaves the probability distribution unchanged.

    GIVEN: a gaussian wavefunction and random potential
    WHEN: I apply the function potential_operator
    THEN: the probability distribution stays unchanged.
    '''

    # set initial state
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    start_position = 0.4
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)
    probability = np.abs(wavefunction) ** 2

    # set potential
    np.random.seed(1)
    potential = np.random.rand(n)

    # set timestep
    dt = 1e-7

    # squared module and integral along x of new wavefunction
    wavefunction_new = wp.potential_operator(wavefunction, potential, dt)
    probability_new = np.abs(wavefunction_new) ** 2
    assert np.all(np.isclose(probability, probability_new))


# kinetic_operator

def test_kinetic_operator_on_probability_distribution():
    '''
    Test if the kinetic operator leaves the probability distribution unchanged.

    GIVEN: the fourier transform of a gaussian wavefunction
    WHEN: I apply the function kinetic_operator
    THEN: the probability distribution stays unchanged.
    '''

    # set real and reciprocal space
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

    # set initial state and fft
    start_position = 0.4
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)
    wavefunction_transform = np.fft.fft(wavefunction)
    transform_probability = 1 / (2 * np.pi * n ** 2) * np.abs(wavefunction_transform) ** 2

    # set timestep
    dt = 1e-7

    # squared module and integral along k of new probability distribution
    wavefunction_transform_new = wp.kinetic_operator(wavefunction_transform, k, dt)
    transform_probability_new = 1 / (2 * np.pi * n ** 2) * np.abs(wavefunction_transform_new) ** 2
    assert np.all(np.isclose(transform_probability, transform_probability_new))


# timestep

def test_timestep_zero_dt():
    '''
    Test if the function timestep leaves the wavefunction unchanged when dt = 0.

    GIVEN: a gaussian wavefunction as initial state
    WHEN: I apply the evolution operator with timestep = 0
    THEN: the wavefunction is unchanged
    '''
    # set real and reciprocal space
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

    # set inital state
    start_position = 0.4
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)

    # set potential
    np.random.seed(1)
    potential = np.random.rand(n)

    # set timestep
    dt = 0

    # apply evolution operator
    wavefunction_new = wp.timestep(wavefunction, potential, k, dt)

    assert np.all(np.isclose(wavefunction, wavefunction_new))


def test_timestep_is_unitary():
    '''
    Test if the evolution operator leaves the norm equal to 1.

    GIVEN: a normalized gaussian wavefunction
    WHEN: I apply the evolution operator for a timestep dt
    THEN: the norm is still equal to 1
    '''
    # set real and reciprocal space
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

    # set inital state
    start_position = 0.4
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)

    # set potential
    np.random.seed(1)
    potential = np.random.rand(n)

    # set timestep
    dt = 1e-7

    # apply evolution operator and compute integral along x
    wavefunction_new = wp.timestep(wavefunction, potential, k, dt)
    probability_new = np.abs(wavefunction_new) ** 2
    I = np.trapz(probability_new, x)

    assert np.isclose(I, 1)


def test_timestep_shift():
    '''
    Test if the shift of the probability distribution equals the group velocity * dt
    when the evolution operator is applied.

    GIVEN: a gaussian wavefunction as initial state and a flat potential
    WHEN: I apply the evolution operator for the timestep dt
    THEN: the probability distribution center shifts to an amount equal to the 
          group velocity * dt
    '''
    # set real and reciprocal space
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

    # set inital state
    start_position = 0.4
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)
    probability = np.abs(wavefunction) ** 2

    # set potential
    potential = np.zeros(n)

    # set timestep
    dt = 1e-7

    # apply evolution operator
    wavefunction_new = wp.timestep(wavefunction, potential, k, dt)
    probability_new = np.abs(wavefunction_new) ** 2

    # compute average position before and after
    x_mean = np.mean(x * probability)
    x_mean_new = np.mean(x * probability_new)

    assert np.isclose(x_mean_new - x_mean, start_momentum * dt)


def test_timestep_dispersion():
    '''
    Test if a gaussian wavepacket spreads when the evolution operator
    is applied.

    GIVEN: a gaussian wavepacket made of different frequencies
    WHEN: the evolution operator is applied
    THEN: the wavepacket spreads.
    '''
    # set real and reciprocal space
    n = 10000
    dx = 1 / n
    x = np.linspace(0, 1, n, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)

    # set inital state
    start_position = 0.4
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)
    probability = np.abs(wavefunction) ** 2

    # set potential
    potential = np.zeros(n)

    # set timestep
    dt = 1e-7

    # apply evolution operator
    wavefunction_new = wp.timestep(wavefunction, potential, k, dt)
    probability_new = np.abs(wavefunction_new) ** 2

    # compute average posi
    x_mean = np.mean(x * probability)
    x_rms = np.sqrt(np.mean(x**2 * probability) - x_mean ** 2)

    x_mean_new = np.mean(x * probability_new)
    x_rms_new = np.sqrt(np.mean(x**2 * probability_new) - x_mean_new ** 2)

    assert x_rms_new > x_rms


def test_timestep_identity():
    '''
    Test if applying the evolution operator for dt and then -dt returns the 
    original wavefunction.
    
    GIVEN: a gaussian wavefunction
    WHEN: I apply the evolution operator for dt and then for -dt
    THEN: I obtain the starting wavefunction
    '''
    # set real and reciprocal space
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

    # set inital state
    start_position = 0.4
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)

    # set potential
    potential = np.zeros(n)

    # set timestep
    dt = 1e-7

    # compute wavefunction after dt and -dt
    wavefunction_forward = wp.timestep(wavefunction, potential, k, dt)
    wavefunction_backward = wp.timestep(wavefunction_forward, potential, k, -dt)

    assert np.all(np.isclose(wavefunction, wavefunction_backward))


def test_timestep_harmonic_groundstate():
    '''
    Test if in the ground state of the harmonic oscillator the probability
    distribution is left unchanged after applying the evolution operator.

    GIVEN: the harmonic oscillator ground state
    WHEN: I apply the evolution operator for dt
    THEN: the probability distribution is left unchanged
    '''


    # set real and reciprocal space
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

    # set inital state
    start_position = 0.5
    sigma = 0.0334370152488211
    start_momentum = 0
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)
    probability = np.abs(wavefunction) ** 2

    # set potential
    potential = wp.harmonic_potential(x, 1e5)

    # set timestep
    dt = 1e-7

    # compute wavefunction after dt and -dt
    wavefunction_new = wp.timestep(wavefunction, potential, k, dt)
    probability_new = np.abs(wavefunction_new) ** 2

    assert np.all(np.isclose(probability, probability_new))


def test_timestep_transform():
    '''
    Test if apply the evolution operator on a wavefunction on a flat potential
    leaves the momentum distribution unchanged.

    GIVEN: a gaussian wavefunction
    WHEN: I apply the evolution operator
    THEN: the momentum distribuiton is left unchanged
    '''
    # set real and reciprocal space
    n = 10000
    x = np.linspace(0, 1, n, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

    # set inital state
    start_position = 0.5
    sigma = 0.01
    start_momentum = 1000
    wavefunction = wp.gaussian_initial_state(x, start_position, sigma, start_momentum)
    wavefunction_transform = np.fft.fft(wavefunction)
    transform_probability = 1 / (2 * np.pi * n ** 2) * np.abs(wavefunction_transform) ** 2

    # set potential
    potential = np.zeros(n)

    # set timestep
    dt = 1e-7

    # compute wavefunction after dt and -dt
    wavefunction_new = wp.timestep(wavefunction, potential, k, dt)
    wavefunction_transform_new = np.fft.fft(wavefunction_new)
    transform_probability_new = 1 / (2 * np.pi * n ** 2) * np.abs(wavefunction_transform_new) ** 2

    assert np.all(np.isclose(transform_probability, transform_probability_new))


def test_barrier_potential_is_symmetric():
    '''
    Test if the function barrier_potential returns a symmetric potential.

    GIVEN: the real space x
    WHEN: I define a barrier potential on it using the function barrier_potential
    THEN: the potential array must be symmetric.
    '''
    n = 1001
    x = np.linspace(0, 1, n, endpoint=False)

    potential = wp.barrier_potential(x, 0.2, 100)

    assert np.all(potential == potential[::-1])


def test_harmonic_potential_is_symmetric():
    '''
    Test if the function harmonic_potential returns a symmetric potential.

    GIVEN: the real space x
    WHEN: I define a harmonic potential on it using the function harmonic_potential
    THEN: the potential array must be symmetric.
    '''
    n = 1001
    x = np.linspace(0, 1, n, endpoint=False)

    potential = wp.harmonic_potential(x, 1e6)

    assert np.all(np.isclose(potential, potential[::-1]))
