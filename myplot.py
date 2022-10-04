import matplotlib.pyplot as plt
import numpy as np

# parameters (to be removed)

n, m = np.loadtxt('input', usecols=2, unpack=True)
n = int(n)
m = int(m)

# spaces (to be removed)
x = np.linspace(0., 1., n, endpoint=False)
k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

# load data
psi_2 = np.load('psi_2.npy')
phi_2 = np.load('phi_2.npy')

# plot
plt.switch_backend('macosx')
plt.ion()

freq = 100
fig, (ax1, ax2) = plt.subplots(2, 1)

for i in range(0, m, freq):
    ax1.clear()
    ax2.clear()
    
    ax1.plot(x, psi_2[:, i])
    ax2.plot(k, phi_2[:, i])
    
    plt.pause(0.01)
    

plt.ioff()
plt.show()
