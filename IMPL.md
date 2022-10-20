# Implementation

The real space consists of $n = 1024$ points from $0$ to $1 - dx$. This can be easily done using the function **np.linspace**.
The time-space consists of $m+1$ points, where $m = int(T // dt)$. $T$ is the simulation time chosen by the user, $dt$ is fixed to be $1e-7$. 

A 2d-array of size $n \times (m+1)$ represents the wave-function during the whole simulation. One column represents the wave-function at one specific time-step.
A second 2d-array represents the Fourier transform after each time-step.

To execute the algorithm quite fast the code relies on NumPy vectorization and the FFT algorithm from the library **np.fft**.

At each time-step, statistical quantities like average position, probability and standard deviation are computed using NumPy dedicated functions.