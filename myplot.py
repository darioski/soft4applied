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

# plotting parameters
freq = int(params.play_speed)

# reduce array size to increase speed
psi_2 = psi_2[:, ::freq]
m = psi_2.shape[1]

y_lim = 1.05 * np.max(psi_2)

# scaling harmonic potential
if params.potential == 'harmonic':
    pot *= 2 * y_lim / params.a


# ----------- animation --------------

fig, ax = plt.subplots()

line, = ax.plot(x, psi_2[:, 0])
ax.plot(x, pot)

time_text = ax.text(0.8, 1.01, 't = {:2.6f}'.format(0), transform=ax.transAxes)

ax.set_xlim(0., 1.)
ax.set_ylim(-0.01*y_lim, y_lim)

def update(frame):
    y = psi_2[:, frame]
    line.set_ydata(y)
    time_text.set_text('t = {:2.6f}'.format(1e-7 * frame * freq))

anim = animation.FuncAnimation(fig, update, frames=m, interval=20)

if params.file_format == 'gif':
    gif_writer = animation.PillowWriter(fps=50)
    anim.save('anim.gif', writer=gif_writer)

# conda install ffmpeg

if params.file_format == 'html':
    with open('anim.html', 'w') as f:
        print(anim.to_html5_video(), file=f)

if params.file_format == 'mp4':
    video_writer = animation.FFMpegWriter(fps=50)
    anim.save('anim.mp4', writer=video_writer)