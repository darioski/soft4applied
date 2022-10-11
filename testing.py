import myfunctions as mf
import numpy as np
import pytest


def test_time_length_1():
    assert mf.check_time_length(1e-7, 1e-7) == None

def test_time_length_2():
    assert mf.check_time_length(0.1, 1e-7) == None

def test_time_length_3():
    assert mf.check_time_length(2, 1e-7) == None

def test_time_length_4():
    with pytest.raises(ValueError):
        mf.check_time_length(9.9e-8, 1e-7)

def test_range_1():
    with pytest.raises(ValueError):
        mf.is_in_range(0.)

def test_range_2():
    with pytest.raises(ValueError):
        mf.is_in_range(1.)

def test_range_3():
    with pytest.raises(ValueError):
        mf.is_in_range(1.1)

def test_range_4():
    with pytest.raises(ValueError):
        mf.is_in_range(-0.1)

def test_range_5():
    assert mf.is_in_range(0.2) == None

def test_range_6():
    assert mf.is_in_range(0.0000001) == None

def test_width_1():
    assert mf.is_wide_enough(3 / 1024, 1/1024) == None
    
def test_width_2():
    assert mf.is_wide_enough(1000, 1/1024) == None

def test_width_3():
    with pytest.raises(ValueError):
        mf.is_wide_enough(1/1024, 1/1024)

def test_width_4():
    with pytest.raises(ValueError):
        mf.is_wide_enough(0, 1/1024)

def test_width_5():
    with pytest.raises(ValueError):
        mf.is_wide_enough(0.0029296874, 1/1024)

def test_width_6():
    with pytest.raises(ValueError):
        mf.is_wide_enough(-0.1, 1/1024)

def test_is_centered_1():
    assert mf.is_centered(0.5, 0.0833) == None

def test_is_centered_3():
    assert mf.is_centered(0., 0.) == None

def test_is_centered_3():
    assert mf.is_centered(0.1, 0.) == None

def test_is_centered_4():
    assert mf.is_centered(18 / 1024, 3 / 1024) == None
    
def test_is_centered_5():
    with pytest.raises(ValueError):
        mf.is_centered(0.3, 0.05)

def test_is_centered_6():
    with pytest.raises(ValueError):
        mf.is_centered(0.5, 0.8333334)

def test_is_centered_7():
    with pytest.raises(ValueError):
        mf.is_centered(6, 1)

def test_is_centered_8():
    with pytest.raises(ValueError):
        mf.is_centered(17 / 1024, 3 / 1024)

def test_pot_input_1():
    assert mf.check_potential('flat') == None

def test_pot_input_2():
    assert mf.check_potential('harmonic') == None

def test_pot_input_3():
    assert mf.check_potential('barrier') == None

def test_pot_input_4():
    with pytest.raises(ValueError):
        mf.check_potential('flar')

def test_pot_input_5():
    with pytest.raises(ValueError):
        mf.check_potential('barier')

def test_pot_input_6():
    with pytest.raises(ValueError):
        mf.check_potential('harmoinic')

def test_pot_input_7():
    with pytest.raises(ValueError):
        mf.check_potential('')

def test_bound_input_1():
    assert mf.check_boundary('periodic') == None

def test_bound_input_2():
    with pytest.raises(ValueError):
        mf.check_boundary('peridoc')

def test_bound_input_3():
    with pytest.raises(ValueError):
        mf.check_boundary('')

def test_gaussian_norm():
    x = np.linspace(0., 1., 1024, endpoint=False)
    i = np.trapz(mf.gaussian(x, 0.5, 0.01), x)
    assert np.isclose(i, 1.)




