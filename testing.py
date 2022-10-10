import myfunctions as mf
import numpy as np
import pytest


def test_gaussian_norm():
    x = np.linspace(0., 1., 1000, endpoint=False)
    i = np.trapz(mf.gaussian(x, 0.5, 0.01), x)
    assert np.isclose(i, 1.)


def test_time_length():
    with pytest.raises(ValueError):
        mf.check_simulation_time(1e-8, 1e7)


def test_start_condition():
    with pytest.raises(ValueError):
        mf.check_start_condition(1., 0.001, 0.01)