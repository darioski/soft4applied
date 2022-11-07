# List of input parameters 
All quantities are expressed in atomic units.

## Simulation settings

* **t _(float)_** _ : Simulation time. 
* **dt _(float)_** _ : Time-step length. 
* **dx _(float)_** _ : Spacial precision. 

## Initial state settings

* **sigma _(float)_** : Initial rms. Cannot be lower than _3 * dx_.
* **start_position _(float)_** : Initial average position. It has to be between 0 and 1.
* **start_momentum _(float)_** : Initial average momentum. 

## Potential profile type

* **potential_type _(str)_** : Potential type.
  * _'flat'_ : Flat potential. Value is set to 0 everywhere.
  * _'barrier'_ : Barrier potential centered in _x = 0.5_.
  * _'delta'_ : Delta-like potential barrier. Value is set to _alpha_ in _x = 0.5_.
  * _'harmonic'_ : Harmonic potential centered in _x = 0.5_.


## Barrier potential
* **half_width _(float)_** : Half-width of the potential barrier. 
* **height _(float)_** : Height of the barrier. If _h < 0_ : potential well.

## Harmonic potential
* **aperture _(float)_** : Aperture of the parabola.

## Delta potential
* **alpha _(float)_** : Height of the delta-like barrier.

## Paths to files 
Paths to the folders where to write the data

## Animation play speed
* **play_speed _(int)_** : Reduce the arrays by this factor to increase the animation speed.

