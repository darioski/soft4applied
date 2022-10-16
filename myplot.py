import matplotlib.pyplot as plt
import numpy as np
import params
import matplotlib.animation as animation


def main():

    n = 1024
    x = np.linspace(0., 1., n, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(n, d=1/n)

    # load data
    with open('pot.npy', 'rb') as f:
        pot = np.load(f)
    with open('psi_2.npy', 'rb') as f:
        psi_2 = np.load(f)

    with open('x_mean.npy', 'rb') as f:
        x_mean = np.load(f)
    with open('x_rms.npy', 'rb') as f:
        x_rms = np.load(f)

    with open('phi_2.npy', 'rb') as f:
        phi_2 = np.load(f)

    # plotting parameters
    freq = int(params.play_speed)

    # reduce array size to increase speed
    psi_2 = psi_2[:, ::freq]
    x_mean = x_mean[::freq]
    x_rms = x_rms[::freq]

    phi_2 = phi_2[:, ::freq]
    m = psi_2.shape[1]

    y_lim = 1.05 * np.max(psi_2)


    # scaling harmonic potential
    if params.potential == 'harmonic':
        pot *= 2 * y_lim / np.abs(params.a)


    # ----------- animation --------------

    fig, ax = plt.subplots()

    line, = ax.plot(x, psi_2[:, 0])
    ax.plot(x, pot)

    time_text_t = ax.text(0.8, 0.95, 't = {:2.6f}'.format(0), transform=ax.transAxes)
    time_text_mean = ax.text(0.4, 0.95, '<x> = {:2.3f}'.format(0), transform=ax.transAxes)
    time_text_rms = ax.text(0.01, 0.95, 'rms(x) = {:2.4f}'.format(0), transform=ax.transAxes)
    
    ax.set_xlim(0., 1.)
    ax.set_ylim(-0.01* y_lim, y_lim)

    def update(frame):
        y = psi_2[:, frame]
        line.set_ydata(y)
        time_text_t.set_text('t = {:2.6f}'.format(1e-7 * frame * freq))
        time_text_mean.set_text('<x> = {:2.3f}'.format(x_mean[frame]))
        time_text_rms.set_text('rms(x) = {:2.4f}'.format(x_rms[frame]))

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

    k = np.concatenate((k[n//2:], k[:n//2]))
    phi_2 = np.concatenate((phi_2[n//2:], phi_2[:n//2]))

    fig, ax = plt.subplots()

    line, = ax.plot(k, phi_2[:, 0])

    time_text_t = ax.text(0.8, 0.95, 't = {:2.6f}'.format(0), transform=ax.transAxes)



    def update(frame):
        y = phi_2[:, frame]
        line.set_ydata(y)
        time_text_t.set_text('t = {:2.6f}'.format(1e-7 * frame * freq))


    anim = animation.FuncAnimation(fig, update, frames=m, interval=20)

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()