# Mass spring damper system

A spring-mass-damping system can be modeled using a mass (m), a spring 
with a constant (k), and a damper with a damping coefficient (b). The 
spring force is proportional to the displacement of the mass, and the 
damping force is proportional to the velocity of the mass.

The equation of motion for the mass-spring-damper system is given by:

$m \ddot{x} + b \dot{x} + k x = F(t)$

where:

- $x$ is the displacement of the mass,

- $\dot{x}$ is the velocity of the mass,

- $\ddot{x}$ is the acceleration of the mass,

- $F(t)$ is the external force applied to the system.

To determine the state-space representation of the mass-spring-damper 
system, we reduce the second-order differential equation to a set of two
 first-order differential equations. We choose the position and velocity
 as our state variables:

$x_1 = x$ y $x_2 = \dot{x}$

The state equations become:

$\dot{x}_1 = x_2$

$\dot{x}_2 = - \frac{k}{m} x_1 - \frac{b}{m} x_2 + \frac{1}{m} F(t)$

The state-space can be represented by:

$A = \begin{bmatrix} 0 & 1 \\ -\frac{k}{m} & -\frac{b}{m} \end{bmatrix}$

$B = \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix}$

$C = \begin{bmatrix} 1 & 0\end{bmatrix}$

The state equation can be written as

$\dot{x} = \begin{bmatrix} \dot{x} \\ \ddot{x} \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -\frac{k}{m} & -\frac{b}{m} \end{bmatrix} \begin{bmatrix} x \\ \dot{x} \end{bmatrix} + \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix} F(t)$

## Simulation

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

b = 4.0  # Damping constant
k = 2.0  # Stiffness of the spring
m = 20.0  # Mass
# F = 5.0  # Force
Ac = np.array([[0.0, 1.0], [-k / m, -b / m]])
Bc = np.array([[0.0], [1.0 / m]])
Cc = np.array([[1.0, 0.0]])
Dc = 0

dt = 0.1  # seconds      

x0 = np.array([0.0, 0.0])
# x0 = np.array([-2.5, 0.25])
(Ad, Bd, Cd, Dd, _) = signal.cont2discrete((Ac, Bc, Cc, Dc), dt, method='zoh', alpha=None)

print(Ad, Bd, Cd, Dd)

Nsim = 1000

Nx = 2
Ny = 1
Nu = 1

Xsim = np.zeros((Nsim+1, Nx))
Ysim = np.zeros((Nsim, Ny))
Ymeas = np.zeros((Nsim, Ny))
Usim = np.ones((Nsim, Nu)) * 0.0
Usim[500:] = 5.0

Xsim[0, :] = x0
for k in range(Nsim):
    Xsim[k+1, :] = Ad @ Xsim[k, :] + Bd @ Usim[k, :]
    Ysim[k, :] = Cd @ Xsim[k, :]
    Ymeas[k, :] = Ysim[k, :] + np.random.randn(Ny) * 0.15

plt.figure()
plt.suptitle("Mass spring damper simulation")

plt.subplot(311)
plt.plot(Xsim[:, 0], label=r"$x_1$", marker='o')
plt.grid()
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Position [m]")

plt.subplot(312)
plt.plot(Xsim[:, 1], label=r"$x_2$", marker='o')
plt.grid()
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Velocity [m/s]")

plt.subplot(313)
plt.plot(Usim[:, 0], label=r"$F(t)$", marker='o')
plt.grid()
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Applied Force [N]")

plt.figure()
plt.plot(Ymeas, label=r"$y$")
plt.plot(Ysim, label=r"$\hat{y}$", alpha=1.0, linewidth=5.0)
plt.grid()
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Measured distance [m]")

plt.show()

# Save data for estimation
np.save("01_Ymeas", Ymeas)
np.save("01_Ysim", Ysim)
np.save("01_Usim", Usim)
np.savez("sys_msd_discrete", Ad=Ad, Bd=Bd, Cd=Cd, Dd=Dd, dt=dt)

```

## References

- [Mechanics Problems using StateSpace](https://docs.sympy.org/latest/tutorials/physics/control/mechanics_problems.html)
- [scipy.signal.cont2discrete](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cont2discrete.html)