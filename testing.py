import wavepacket as wp
import numpy as np
import pytest

dt = 1e-7
n = 1024
x = np.linspace(0., 1., n, endpoint=False)
k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)


# initial_state
def test_initial_state_1():
    psi = wp.initial_state(x, 0.5, 0.01, 2000)
    assert psi.dtype == 'complex128'
    
def test_initial_state_2(): 
    psi = wp.initial_state(x, 0.5, 0.01, 2000)
    i = np.trapz(np.abs(psi) ** 2, x)
    assert np.isclose(i, 1)


# potential_operator
def test_pot_op_1():
    psi = wp.initial_state(x, 0.5, 0.01, 2000)
    pot = np.zeros(n)
    assert wp.potential_operator(psi, pot, dt).dtype == 'complex128'

def test_pot_op_2():
    psi = wp.initial_state(x, 0.1, 0.01, 2000)
    pot = np.zeros(n)
    assert np.all(wp.potential_operator(psi, pot, dt) == psi)

def test_pot_op_3():
    psi = wp.initial_state(x, 0.1, 0.01, 2000)
    pot = wp.harmonic_potential(x, 1e8)
    i = np.trapz(np.abs(psi)**2, x)
    j = np.trapz(np.abs(wp.potential_operator(psi, pot, dt))**2, x)
    assert np.isclose(i, j)

# kinetic_operator
def test_kin_op_1():
    psi = wp.initial_state(x, 0.5, 0.01, 2000)
    phi = np.fft.fft(psi)
    assert wp.kinetic_operator(phi, k, dt).dtype == 'complex128'


def test_kin_op_2():
    psi = wp.initial_state(x, 0.2, 0.01, 2000)
    phi = np.fft.fft(psi)
    i = 1 / (2 * np.pi * n ** 2) * np.trapz(np.abs(phi) ** 2, k)
    assert np.isclose(i, 1)
    


