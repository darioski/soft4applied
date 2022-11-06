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
