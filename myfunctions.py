import numpy as np
import pytest

def initial_conditions(x):

    sigma = 1
    a = 1 / (2 * np.pi * sigma ** 2) ** 0.25    # normalization
    x_0 = 10    # initial position
    k_0 = 20    # initial momentum

    psi = a * np.exp(1j * k_0 * x - ((x - x_0) / (2 * sigma)) ** 2)

    return psi


# ----------- tests ------------

def test_empty():
    x = np.array([])
    assert len(initial_conditions(x)) == 0
