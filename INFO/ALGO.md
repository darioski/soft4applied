# The algorithm

The hamiltonian of the problem can be divided into a kinetic and a potential part: $\hat{H} = \hat{T} + \hat{V}$.

Therefore, the unitary evolution operator can be separated as well, with little approximation:

$\hat{U}(t) = e^{-\frac{i}{\hbar} \hat{T} \frac{t}{2}} e{-\frac{i}{\hbar} \hat{V} t} + O(t^2)$

We can do even better using the Suzuki-Trotter expansion at the second order:

$\hat{U}(t) = e{-\frac{i}{\hbar} \hat{V} \frac{t}{2}} e^{-\frac{i}{\hbar} \hat{T} \frac{t}{2}} e{-\frac{i}{\hbar} \hat{V} \frac{t}{2}} + O(t^3)$

Applying the potential part is a simple element-by-element multiplication: $e{-\frac{i}{\hbar} \hat{V}(x) \frac{t}{2}} \psi(x)$.

To make it a simple multiplication also for the kinetic part, we have to transform the wave-function to the reciprocal space using the **Fourier transform**: $\phi(k) = F(\psi(x))$.

Applying the kinetic evolution operator now simply means performing the element-by-element multiplication: $e{-\frac{i}{\hbar} -k^2 \frac{t}{2}} \phi(k)$.

To sum up, a single timestep of the algorithm needs:
1. Apply the operator  $e{-\frac{i}{\hbar} \hat{V}(x) \frac{t}{2}}$ to the original function $\psi(T)$.
2. Transform the resulting function to the reciprocal space.
3. Apply the operator $e{-\frac{i}{\hbar} -k^2 \frac{t}{2}}$ to the intermediate function.
4. Transform the result back to the real space using the inverse Fourier transform.
5. Apply again the operator  $e{-\frac{i}{\hbar} \hat{V}(x) \frac{t}{2}}$ to finally obtain $\psi(T + t)$.