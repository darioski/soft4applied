# List of input parameters (_params.py_)
All quantities are expressed in atomic units.

## Simulation time

* **t _(float)_** _ : Simulation time. Has to be bigger than _dt = 1e-7_.

## Initial state

* **sigma _(float)_** : Initial rms. Cannot be lower than _3 * dx_, where _dx = 1/1024_.
* **x_0 _(float)_** : Initial position of the gaussian. It has to be between 0 and 1.
* **k_0 _(float)_** : Initial momentum. It has to be lower than _2 * np.pi * NF_, where _NF_ is the Nyquist frequency.


## Boundary conditions

* **boundary _(str)_** : Boundary conditions. 
  * _'periodic'_ : Periodic boundary conditions.

## Potential

* **potential _(str)_** : Potential type.
  * _'flat'_ : Flat potential. Value is set to 0 everywhere.
  * _'barrier'_ : Barrier potential centered in _x = 0.5_.
    * **b _(float)_** : Half-width of the potential barrier. Total width = _2 * b_.
    * **h _(float)_** : Height of the barrier. If _h < 0_ : potential well.
  * _'delta'_ : Delta-like potential barrier. Value is set to _alpha_ in _x = 0.5_.
    * **alpha _(float)_** : Height of the delta-like barrier.
  * _'harmonic'_ : Harmonic potential centered in _x = 0.5_.
    * **a _(float)_** : Aperture of the parabola _a * x ** 2_.

## Plot settings

* **file_format _(str)_** : Save the animation in the desired format.
  Possible choices are: _'gif'_, _'mp4'_.

* **play_speed _(int)_** : Reduce the arrays by this factor to increase the animation speed.

