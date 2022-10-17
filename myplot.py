import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import params
import pandas as pd


# load data
with open('pot.npy', 'rb') as f:
    pot = np.load(f)
with open('psi_2.npy', 'rb') as f:
    psi_2 = np.load(f)
with open('phi_2.npy', 'rb') as f:
    phi_2 = np.load(f)


stat = pd.read_csv('statistics.csv')
p_left = np.array(stat['p_left'])
x_mean = np.array(stat['x_mean'])
x_rms = np.array(stat['x_rms'])
k_mean = np.array(stat['k_mean'])
k_rms = np.array(stat['k_rms'])


# reduce array size to increase speed
freq = int(params.play_speed)
psi_2 = psi_2[:, ::freq]
p_left = p_left[::freq]
x_mean = x_mean[::freq]
x_rms = x_rms[::freq]
phi_2 = phi_2[:, ::freq]
k_mean = k_mean[::freq]
k_rms = k_rms[::freq]

m = psi_2.shape[1]
n = len(pot)
x = np.linspace(0., 1., n, endpoint=False)
k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

y_lim =  1.2 * np.max(psi_2[:, 0])


# scaling harmonic potential
if params.potential == 'harmonic':
    pot *= 2 * y_lim / abs(params.a)

k = np.concatenate((k[n//2:], k[:n//2]))
phi_2 = np.concatenate((phi_2[n//2:], phi_2[:n//2]))


# ----------- animation --------------
gs_kw = {'height_ratios':[0.7, 0.3]}

fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw=gs_kw)

ax1.set_xlim(0., 1.)
ax1.set_ylim(-0.01* y_lim, y_lim)
ax1.grid(ls='--')
line1, = ax1.plot(x, psi_2[:, 0], label=r'$|\Psi$(x)|$^2$')
ax1.plot(x, pot, label='V(x)')

ax1.legend(loc=(0.8, 0.72), frameon=False)

time_text_t = ax1.text(0.8, 0.94, 't={:2.6f}'.format(0), transform=ax1.transAxes)
time_text_left = ax1.text(0.52, 0.94, 't={:2.4f}'.format(0), transform=ax1.transAxes)
time_text_mean = ax1.text(0.01, 0.94, '<x>={:2.3f}'.format(0), transform=ax1.transAxes)
time_text_rms = ax1.text(0.24, 0.94, 'rms(x)={:2.4f}'.format(0), transform=ax1.transAxes)


ax2.grid(ls='--')
line2, = ax2.plot(k, phi_2[:, 0], label=r'$|\Phi$(k)|$^2$')
ax2.legend(frameon=False)

time_text_kmean = ax2.text(0.01, 0.85, '<k>={:3.0f}'.format(0), transform=ax2.transAxes)
time_text_krms = ax2.text(0.3, 0.85, 'rms(k)={:3.0f}'.format(0), transform=ax2.transAxes)

def update(frame):
    line1.set_ydata(psi_2[:, frame])
    time_text_t.set_text('t = {:2.6f}'.format(1e-7 * frame * freq))
    time_text_left.set_text('p(x<0.5)= {:2.4f}'.format(p_left[frame]))
    time_text_mean.set_text('<x> = {:2.3f}'.format(x_mean[frame]))
    time_text_rms.set_text('rms(x) = {:2.4f}'.format(x_rms[frame]))

    line2.set_ydata(phi_2[:, frame])
    time_text_kmean.set_text('<k>={:3.0f}'.format(k_mean[frame]))
    time_text_krms.set_text('rms(k)={:3.0f}'.format(k_rms[frame]))

anim = animation.FuncAnimation(fig, update, frames=m, interval=20)

# if params.file_format == 'gif':
#     gif_writer = animation.PillowWriter(fps=50)
#     anim.save('anim.gif', writer=gif_writer)

# # conda install ffmpeg

# if params.file_format == 'html':
#     with open('anim.html', 'w') as f:
#         print(anim.to_html5_video(), file=f)

# if params.file_format == 'mp4':
#     video_writer = animation.FFMpegWriter(fps=50)
#     anim.save('anim.mp4', writer=video_writer)


plt.show()
plt.close()
