import matplotlib.pyplot as plt
import numpy as np
import params
import matplotlib.animation as animation

# load data
with open('x.npy', 'rb') as f:
    x = np.load(f)
with open('pot.npy', 'rb') as f:
    pot = np.load(f)
with open('psi_2.npy', 'rb') as f:
    psi_2 = np.load(f)


# plot
# plotting parameters
m = int(1e7 * params.t)
freq = int(params.freq)
y_lim = 1.05 * np.max(psi_2)

if params.potential == 'harmonic':
    pot *= 2 * y_lim / params.a

if params.potential == 'barrier':
    pot *= 0.95 * y_lim / np.abs(params.h)



# ----------- animation --------------

# plt.show()

# for j in range(0, m, freq):
#     ax.clear()
#     ax.set_ylim(-0.01*y_lim, y_lim)

#     ax.plot(x, psi_2[:, j])
#     ax.plot(x, pot)

#     ax.text(0.8, 1.01, 't = {:2.2e}'.format(1e-7 * j), transform=ax.transAxes)
    
#     plt.pause(0.01)


fig, ax = plt.subplots()

line, = ax.plot(x, psi_2[:, 0])
ax.plot(x, pot)

time_text = ax.text(0.8, 1.01, 't = {:2.2e}'.format(1e-7 * 0), transform=ax.transAxes)

ax.set_xlim(0., 1.)
ax.set_ylim(-0.01*y_lim, y_lim)

def update(frame):
    y = psi_2[:, frame+1]
    line.set_ydata(y)
    time_text.set_text('t = {:2.6f}'.format(1e-7 * frame))


anim = animation.FuncAnimation(fig, update, frames=m, interval=20)

plt.show()
