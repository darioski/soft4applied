import matplotlib.pyplot as plt
import numpy as np

# parameters
le = 50
n = 1024

# x-space
x = np.linspace(0, le, n, endpoint=False)

# load data
psi_2 = np.load('output.npy')

# plot
fig, ax = plt.subplots()

ax.plot(x, psi_2)

plt.show()