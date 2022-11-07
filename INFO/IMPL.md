# Implementation

The real space consists of $n = \frac{1}{dx}$ points from $0$ to $1 - dx$.
The time-space consists of $m+1$ points, where $m = int(T / dt)$. Both $T$ and $dt$ are chosen by the user. 

A 2d-array of size $n \times (m+1)$ represents the wave-function during the whole simulation. One column represents the wave-function at one specific time-step.
A second 2d-array represents the Fourier transform after each time-step.

To execute the algorithm quite fast the code relies on NumPy vectorization and the FFT algorithm from the library **np.fft**.

At each time-step, statistical quantities like average position, probability and standard deviation are computed using NumPy dedicated functions.