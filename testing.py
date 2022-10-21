import wavepacket as wp
import numpy as np
import pytest

dt = 1e-7
n = 1024
x = np.linspace(0., 1., n, endpoint=False)
k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

boundary_list = ['periodic']
potential_list = ['flat', 'barrier', 'harmonic', 'delta']


# check_time_length
def test_time_length_1():
    assert wp.check_time_length(1e-7, 1e-7) == None

def test_time_length_2():
    assert wp.check_time_length(0.1, 1e-7) == None

def test_time_length_3():
    assert wp.check_time_length(2, 1e-7) == None

def test_time_length_4():
    with pytest.raises(ValueError):
        wp.check_time_length(9.9e-8, 1e-7)

def test_range_1():
    with pytest.raises(ValueError):
        wp.is_in_range(0.)

# is_in_range
def test_range_2():
    with pytest.raises(ValueError):
        wp.is_in_range(1.)

def test_range_3():
    with pytest.raises(ValueError):
        wp.is_in_range(1.1)

def test_range_4():
    with pytest.raises(ValueError):
        wp.is_in_range(-0.1)

def test_range_5():
    assert wp.is_in_range(0.2) == None

def test_range_6():
    assert wp.is_in_range(0.0000001) == None

# is_wide_enough
def test_width_1():
    assert wp.is_wide_enough(3 / 1024, 1/1024) == None
    
def test_width_2():
    assert wp.is_wide_enough(1000, 1/1024) == None

def test_width_3():
    with pytest.raises(ValueError):
        wp.is_wide_enough(1/1024, 1/1024)

def test_width_4():
    with pytest.raises(ValueError):
        wp.is_wide_enough(0, 1/1024)

def test_width_5():
    with pytest.raises(ValueError):
        wp.is_wide_enough(0.0029296874, 1/1024)

def test_width_6():
    with pytest.raises(ValueError):
        wp.is_wide_enough(-0.1, 1/1024)

# is_centered
def test_is_centered_1():
    assert wp.is_centered(0.5, 0.0833) == None

def test_is_centered_3():
    assert wp.is_centered(0., 0.) == None

def test_is_centered_3():
    assert wp.is_centered(0.1, 0.) == None

def test_is_centered_4():
    assert wp.is_centered(18 / 1024, 3 / 1024) == None
    
def test_is_centered_5():
    with pytest.raises(ValueError):
        wp.is_centered(0.3, 0.05)

def test_is_centered_6():
    with pytest.raises(ValueError):
        wp.is_centered(0.5, 0.8333334)

def test_is_centered_7():
    with pytest.raises(ValueError):
        wp.is_centered(6, 1)

def test_is_centered_8():
    with pytest.raises(ValueError):
        wp.is_centered(17 / 1024, 3 / 1024)


# check_initial_momentum
def test_init_momentum_1():
    assert wp.check_initial_momentum(n, 0.1, 3100) == None

def test_init_momentum_2():
    assert wp.check_initial_momentum(n, 0.01, -2500) == None

def test_init_momentum_3():
    with pytest.raises(ValueError):
        wp.check_initial_momentum(n, 0.01, 3000)

def test_init_momentum_4(): 
    with pytest.raises(ValueError):
        wp.check_initial_momentum(n, 0.001, 1)

def test_init_momentum_5():
    with pytest.raises(ValueError):
        wp.check_initial_momentum(n, 0.003, 2200)

# check_potential
def test_pot_input_1():
    assert wp.check_potential('flat', potential_list) == None

def test_pot_input_2():
    assert wp.check_potential('harmonic', potential_list) == None

def test_pot_input_3():
    assert wp.check_potential('barrier', potential_list) == None

def test_pot_input_4():
    with pytest.raises(ValueError):
        wp.check_potential('flar', potential_list)

def test_pot_input_5():
    with pytest.raises(ValueError):
        wp.check_potential('barier', potential_list)

def test_pot_input_6():
    with pytest.raises(ValueError):
        wp.check_potential('harmoinic', potential_list)

def test_pot_input_7():
    with pytest.raises(ValueError):
        wp.check_potential('', potential_list)

# check_boundary
def test_bound_input_1():
    assert wp.check_boundary('periodic', boundary_list) == None

def test_bound_input_2():
    with pytest.raises(ValueError):
        wp.check_boundary('peridoc', boundary_list)

def test_bound_input_3():
    with pytest.raises(ValueError):
        wp.check_boundary('', boundary_list)

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
    


