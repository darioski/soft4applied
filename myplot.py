import matplotlib.pyplot as plt
import numpy as np
import pickle
import params

# load data
with open('data.pickle', 'rb') as datafile:
    data = pickle.load(datafile)

x = data['x']
k = data['k']
pot = data['pot']
psi = data['psi']
phi = data['phi']
psi_2 = data['psi_2']
phi_2 = data['phi_2']


# plot
# plotting parameters
m = int(1e7 * params.t)
freq = int(params.freq)
y_lim = 1.05 * np.max(psi_2)

if params.potential == 'harmonic':
    pot *= 2 * y_lim / params.a

if params.potential == 'barrier':
    pot *= 0.95 * y_lim / np.abs(params.h)


plt.switch_backend('macosx')
plt.ion()

fig, ax = plt.subplots()

for j in range(0, m, freq):
    ax.clear()
    ax.set_ylim(-0.01*y_lim, y_lim)

    ax.plot(x, psi_2[:, j])
    ax.plot(x, pot)

    ax.text(0.8, 1.01, 't = {:2.2e}'.format(1e-7 * j), transform=ax.transAxes)
    
    plt.pause(0.01)

plt.ioff()
plt.show()
