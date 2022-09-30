from mimetypes import init
import numpy as np
from myfunctions import *
import pytest

# parameters
le = 50
n = 1024

# x-space
x = np.linspace(0, le, n, endpoint=False)

# wave packet
psi = initial_conditions(x)

# compute squared module
psi_2 = np.abs(psi) ** 2

# save in output file
with open('output.npy', 'wb') as f:
    np.save(f, psi_2)

# --------- tests ----------

def test_length():
    assert len(psi) == n

def test_type():
    assert psi.dtype == 'complex128'

def test_normalization():
    assert np.trapz(np.abs(psi) ** 2, x) == 1.