# Quantum wave-packet dynamics


## Theory 



See [Theory]

## Code implementation



## Usage
Edit the input file _./params.py_ and choose the value of the desired parameters. See [List of input parameters](./PARAMS.md).

After the input file has been edited, run the simulation with the command
```
python ./run.py
```
The script returns two binary files in the _.npy_ format, where the _Nx(M+1)_ NumPy arrays are stored. 
It also returns a _.csv_ file with the statistics computed for each simulation step.


To visualize the simulation, run the script _./myplot.py_ with the command
```
python ./myplot.py
```
If you want to save the animation, specify the format in the input file _./params.py_.
*Warning:* If you want to save the animation as _.mp4_ or as _.html_, you may need the package _ffmpeg_.
If this is the case, install it by running on your terminal the command `conda install ffmpeg`.

## Examples

Here are some examples simulating known quantum phenomena:

* [spreading]
* [potential well]
* [quantum tunneling]
* [harmonic oscillator]

