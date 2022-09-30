import matplotlib.pyplot as plt
import numpy as np

# parameters (to be removed)
le = 50
n = 1024

# spaces (to be removed)
x = np.linspace(0, le, n, endpoint=False)
k = 2 * np.pi * np.fft.fftfreq(n, d=le/n)

# load data
psi_2 = np.load('psi_2.npy')
phi_2 = np.load('phi_2.npy')

# plot
fig, ax = plt.subplots(2, 1)

ax[0].plot(x, psi_2)
ax[1].plot(k, phi_2)

plt.show()