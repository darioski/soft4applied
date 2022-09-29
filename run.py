import numpy as np
from myfunctions import *



# global variables
le = 50     # length
n = 1024    # number of points
x_0 = 10    # intial position

m = 2000    # time sampling


with open('input', 'r') as file:
    sigma = float(file.readline())
    kappa = float(file.readline())


x = xspace(le, n)
#k = kspace(le, n)

# 
psi = np.zeros((n, m+1), dtype=complex)
phi = np.zeros((n, m+1), dtype=complex)

initial_conditions(sigma, kappa, psi, x, x_0)

# write output wave
with open('output', 'w') as file:
    file.write(psi[:, 0])