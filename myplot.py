import matplotlib.pyplot as plt
import numpy as np
import pickle

# parameters (to be removed)

n, m = np.loadtxt('input', usecols=2, unpack=True)
n = int(n)
m = int(m)

# spaces (to be removed)
x = np.linspace(0., 1., n, endpoint=False)
k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

# pot
pot = np.zeros(n)

# load data
with open('data.pickle', 'rb') as datafile:
    data = pickle.load(datafile)

psi = data['psi']
phi = data['phi']
psi_2 = data['psi_2']
phi_2 = data['phi_2']


# plot
plt.switch_backend('macosx')
plt.ion()

freq = 1
fig, (ax1, ax2) = plt.subplots(2, 1)

for j in range(0, m, freq):
    ax1.clear()
    ax2.clear()
    
    ax1.set_ylim(0, 1)

    ax1.plot(x, psi_2[:, j], x, pot)
    ax2.plot(k, phi_2[:, j])
    
    plt.pause(0.01)
    

plt.ioff()
plt.show()
