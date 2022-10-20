# Quantum wave-packet dynamics


## Theory 
The [time-dependent Schrödinger equation](https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation) describes the dynamics of a particle as a wave. The wave-function is indicated with the symbol $\psi(x, t)$. A **unitary evolution operator** $U(t) = e^{-\frac{i}{\hbar} \hat{H} t}$ can be defined so that $\psi(x, t) = \hat{U}(t) \psi(x, 0)$, which satisfies the equation.

Given as initial condition $\psi(x, 0)$ a **gaussian wave-packet** with initial average momentum $k_0$, it is possible to numerically solve the problem using small time-steps. See [Algorithm](./INFO/ALGO.md).

## Code implementation
The code exploits the library [NumPy](https://numpy.org/) and the speed it provides when operating
with matrices. For details, see [Code implementation](./INFO/IMPL.md).


## Usage
1. Edit the input file _./params.py_ and choose the value of the desired parameters. See [List of input parameters](./INFO/PARAMS.md).

2. After the input file has been edited, run the simulation with the command
```
python ./run.py
```
The script returns three binary files in the _.npy_ format, where the two _Nx(M+1)_ NumPy arrays and the potential array are stored. 
It also returns a _.csv_ file with the statistics computed for each step of the simulation.


3. To visualize the simulation, run the script _./myplot.py_ with the command
```
python ./myplot.py
```
If you want to save the animation, specify the format in the input file _./params.py_.

**Warning:** If you want to save the animation as _.mp4_, you may need the package _ffmpeg_.
If this is the case, install it by running on your terminal the command `conda install ffmpeg`.

## Examples

Here are some example simulating quantum behaviour:

* [Wave-packet spreading](examples/spreading.md)
* [Potential well](examples/well.md)
* [Quantum tunneling](examples/tunnel.md)
* [Harmonic oscillator ground state](examples/oscill.md)