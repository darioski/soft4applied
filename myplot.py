import matplotlib.pyplot as plt
import numpy as np
import pickle


# read parameters from file
t, sigma, x_0, k_0, a, h = np.loadtxt('input', usecols=2, unpack=True)
n = 1024
dt = 1e-7
m = int(1e7 * t)

# spaces
x = np.linspace(0., 1., n, endpoint=False)
k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

# load data
with open('data.pickle', 'rb') as datafile:
    data = pickle.load(datafile)

pot = data['pot']
psi = data['psi']
phi = data['phi']
psi_2 = data['psi_2']
phi_2 = data['phi_2']


# plot
plt.switch_backend('macosx')
plt.ion()

freq = 10
fig, ax = plt.subplots()



for j in range(0, m, freq):
    ax.clear()
    ax.set_ylim(-1, 1.2 * 1 / np.sqrt(2 * np.pi * sigma ** 2))

    ax.plot(x, psi_2[:, j])
    ax.plot(x, pot)
    
    plt.pause(0.01)

plt.ioff()
plt.show()
