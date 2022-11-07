import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import configparser
from pathlib import Path

config = configparser.ConfigParser()
config.read('config.txt')

potential = config.get('settings', 'potential_type')
freq = int(config.get('settings', 'play_speed'))

filepath_1 = config.get('paths', 'potential')
filepath_2 = config.get('paths', 'probability')
filepath_3 = config.get('paths', 'transform_probability')
filepath_4 = config.get('paths', 'statistics')

filepath_5 = config.get('paths', 'animation')


# load data to plot
with open(filepath_1, 'rb') as f:
    pot = np.load(f)
with open(filepath_2, 'rb') as f:
    psi_2 = np.load(f)
with open(filepath_3, 'rb') as f:
    phi_2 = np.load(f)

stats = pd.read_csv(filepath_4)

# reduce arrays size to increase speed

psi_2 = psi_2[:, ::freq]
phi_2 = phi_2[:, ::freq]

time = np.array(stats['time'])[::freq]
p_left = np.array(stats['p_left'])[::freq]
x_mean = np.array(stats['x_mean'])[::freq]
x_rms = np.array(stats['x_rms'])[::freq]
pk_left = np.array(stats['pk_left'])[::freq]
k_mean = np.array(stats['k_mean'])[::freq]
k_rms = np.array(stats['k_rms'])[::freq]

# set parameters and spaces for plotting
m = len(p_left)
n = len(pot)
x = np.linspace(0., 1., n, endpoint=False)
k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

# set ylim 
y_lim =  1.1 * np.max(psi_2[:, 0])

# rescale barrier potential to fit in the figure
if potential == 'barrier':
    h = float(config.get('settings', 'height'))
    if h < 0:
        pot = 0.8 * y_lim * pot / abs(h) + 0.85 * y_lim
    else:
        pot *= 0.85 * y_lim / h

# rescale harmonic potential to fit in the figure
if potential == 'harmonic':
    a = float(config.get('settings', 'aperture'))
    y_lim = 1.1 * np.max(psi_2)
    pot *= 2 * y_lim / abs(a)

# invert arrays for better plotting
k = np.concatenate((k[n//2:], k[:n//2]))
phi_2 = np.concatenate((phi_2[n//2:], phi_2[:n//2]))

# ----------- animation --------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={'height_ratios':[0.7, 0.3]})

# real space
ax1.set_xlim(0., 1.)
ax1.set_ylim(-0.01*y_lim, y_lim)
ax1.grid(ls='--')
ax1.set_ylabel('$|\Psi$(x)|$^2$')
line1, = ax1.plot(x, psi_2[:, 0], 'k', lw=1, label='$|\Psi$(x)|$^2$')
ax1.plot(x, pot, lw=1, label='V(x)')

ax1.legend(loc=(0.85, 0.69), frameon=False)

time_text_t = ax1.text(0.8, 0.94, 't={:2.6f}'.format(0), transform=ax1.transAxes)
time_text_left = ax1.text(0.52, 0.94, 'P(x<0.5)={:2.4f}'.format(0), transform=ax1.transAxes)
time_text_mean = ax1.text(0.01, 0.94, '<x>={:2.3f}'.format(0), transform=ax1.transAxes)
time_text_rms = ax1.text(0.24, 0.94, 'rms(x)={:2.4f}'.format(0), transform=ax1.transAxes)

# reciprocal space
ax2.grid(ls='--')
ax2.set_ylabel('$|\Phi$(k)|$^2$')
line2, = ax2.plot(k, phi_2[:, 0], 'r', lw=1, label='$|\Phi$(k)|$^2$')
ax2.legend(loc=(0.85, 0.72), frameon=False)

time_text_kleft = ax2.text(0.67, 0.85, 'P(k<0)={:2.4f}'.format(0), transform=ax2.transAxes)
time_text_kmean = ax2.text(0.01, 0.85, '<k>={:3.0f}'.format(0), transform=ax2.transAxes)
time_text_krms = ax2.text(0.2, 0.85, 'rms(k)={:3.0f}'.format(0), transform=ax2.transAxes)

def update(frame):
    # real space
    line1.set_ydata(psi_2[:, frame])

    time_text_t.set_text('t = {:2.6f}'.format(time[frame]))
    time_text_left.set_text('P(x<0.5)= {:2.4f}'.format(p_left[frame]))
    time_text_mean.set_text('<x> = {:2.3f}'.format(x_mean[frame]))
    time_text_rms.set_text('rms(x) = {:2.4f}'.format(x_rms[frame]))

    # reciprocal space
    line2.set_ydata(phi_2[:, frame])
    time_text_kleft.set_text('P(k<0)= {:2.4f}'.format(pk_left[frame]))
    time_text_kmean.set_text('<k>={:3.0f}'.format(k_mean[frame]))
    time_text_krms.set_text('rms(k)={:3.0f}'.format(k_rms[frame]))

# create animation to show/save
anim = animation.FuncAnimation(fig, update, frames=m, interval=20)

if filepath_5[-4:] == '.gif':
    Path(filepath_5).parent.mkdir(parents=True, exist_ok=True)
    gif_writer = animation.PillowWriter(fps=50)
    anim.save(filepath_5, writer=gif_writer)

if filepath_5[-4:] == '.mp4':
    Path(filepath_5).parent.mkdir(parents=True, exist_ok=True)
    video_writer = animation.FFMpegWriter(fps=50)
    anim.save('anim.mp4', writer=video_writer)

if filepath_5[-4:] not in ['.gif', '.mp4']:
    plt.show()
    plt.close()