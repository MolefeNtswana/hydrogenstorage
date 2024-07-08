# hydrogenstorage
hydrogen-storage

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pyvista as pv

# Define constants
R = 8.314  # Universal gas constant, J/(mol K)
T_amb = 300  # Ambient temperature in Kelvin
P_amb = 101325  # Ambient pressure in Pa
c_p = 4.18  # Specific heat capacity, J/(g K)
m_hydride = 1000  # Mass of the hydride in grams
V = 0.01  # Volume of the tank in m^3
delta_H = -75000  # Enthalpy change in J/mol
delta_S = -130  # Entropy change in J/(mol K)
T_ref = 298  # Reference temperature in K
activation_energy = 50000  # Activation energy in J/mol
rate_constant_a = 0.001  # Rate constant for absorption
rate_constant_d = 0.001  # Rate constant for desorption
k_eff = 0.5  # Effective thermal conductivity in W/(m K)

# Metal hydride properties
capacity = 1500  # Max hydrogen capacity in mol/m^3
K = 0.001  # Permeability for Darcy flow
mu = 8.9e-6  # Dynamic viscosity for turbulent flow, Pa.s
rho = 0.09  # Density of hydrogen gas in kg/m^3

# Define the tank geometry
tank_radius = 0.1  # in meters
tank_height = 1.0  # in meters
tank_volume = np.pi * tank_radius**2 * tank_height

# Simulation parameters
t_final = 1000  # Final time in seconds
dt = 1  # Time step in seconds
time = np.arange(0, t_final, dt)

# Initial conditions
initial_temperature = T_amb
initial_pressure = P_amb
initial_fraction = 0  # Initial fraction of hydrogen absorbed

# Initialize arrays to store results
temperatures = np.full(len(time), initial_temperature)
pressures = np.full(len(time), initial_pressure)
equilibrium_pressures = np.full(len(time), initial_pressure)
velocities_darcy = np.zeros(len(time))
velocities_turbulent = np.zeros(len(time))
velocities_laminar = np.zeros(len(time))
velocities_transition = np.zeros(len(time))
mass_stored_darcy = np.zeros(len(time))
mass_stored_turbulent = np.zeros(len(time))
mass_stored_laminar = np.zeros(len(time))
mass_stored_transition = np.zeros(len(time))
fractions = np.full(len(time), initial_fraction)

# Define functions for flow regimes
def darcy_flow(K, dP, mu, L):
    return -K / mu * dP / L

def turbulent_flow(dP, rho, L):
    return np.sqrt(2 * dP / rho / L)

def laminar_flow(mu, dP, L, r):
    return (r**2 / (8 * mu)) * (dP / L)

def transition_flow(dP, rho, mu, L, r):
    Re = rho * r * abs(dP) / (mu * L)
    if Re < 2000:
        return laminar_flow(mu, dP, L, r)
    elif Re > 4000:
        return turbulent_flow(dP, rho, L)
    else:
        return (laminar_flow(mu, dP, L, r) + turbulent_flow(dP, rho, L)) / 2

# Define the hydrogen absorption kinetics function
def hydrogen_absorption(X, t, T, P, P_eq):
    r_a = rate_constant_a * (P - P_eq) * (1 - X)
    r_d = rate_constant_d * (P_eq - P) * X
    return r_a - r_d

# Simulation loop
for t in range(1, len(time)):
    # Calculate the pressure gradient (simplified)
    dP = pressures[t-1] - P_amb

    # Calculate velocities based on flow regimes
    velocities_darcy[t] = darcy_flow(K, dP, mu, tank_height)
    velocities_turbulent[t] = turbulent_flow(dP, rho, tank_height)
    velocities_laminar[t] = laminar_flow(mu, dP, tank_height, tank_radius)
    velocities_transition[t] = transition_flow(dP, rho, mu, tank_height, tank_radius)

    # Update temperature (considering heat transfer in porous media)
    Q = delta_H * fractions[t-1]
    T_prev = temperatures[t-1]
    convective_term = velocities_darcy[t] * (T_amb - T_prev)
    conductive_term = k_eff * (T_amb - T_prev) / tank_height
    temperatures[t] = T_prev + dt * (Q / (m_hydride * c_p) + convective_term + conductive_term)

    # Calculate equilibrium pressure using Van't Hoff equation
    equilibrium_pressures[t] = np.exp(delta_H / R * (1 / T_ref - 1 / temperatures[t])) * np.exp(delta_S / R)

    # Update hydrogen fraction absorbed using ODE solver
    fractions[t] = odeint(hydrogen_absorption, fractions[t-1], [0, dt], args=(temperatures[t], pressures[t-1], equilibrium_pressures[t]))[-1]

    # Update pressure using the ideal gas law
    pressures[t] = P_amb * (1 - fractions[t])

    # Calculate mass stored considering the mass source term
    r_a = rate_constant_a * (pressures[t-1] - equilibrium_pressures[t]) * (1 - fractions[t-1])
    r_d = rate_constant_d * (equilibrium_pressures[t] - pressures[t-1]) * fractions[t-1]
    mass_source = r_a - r_d
    mass_stored_darcy[t] = mass_stored_darcy[t-1] + mass_source * dt
    mass_stored_turbulent[t] = mass_stored_turbulent[t-1] + mass_source * dt
    mass_stored_laminar[t] = mass_stored_laminar[t-1] + mass_source * dt
    mass_stored_transition[t] = mass_stored_transition[t-1] + mass_source * dt

# Plot results
plt.figure(figsize=(12, 10))

# Plot mass stored comparison
plt.subplot(4, 1, 1)
plt.plot(time, mass_stored_darcy, label='Darcy Flow')
plt.plot(time, mass_stored_turbulent, label='Turbulent Flow')
plt.plot(time, mass_stored_laminar, label='Laminar Flow')
plt.plot(time, mass_stored_transition, label='Transition Flow')
plt.xlabel('Time (s)')
plt.ylabel('Mass Stored (mol)')
plt.title('Hydrogen Mass Stored over Time')
plt.legend()

# Plot temperature evolution
plt.subplot(4, 1, 2)
plt.plot(time, temperatures, label='Temperature')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Evolution over Time')
plt.legend()

# Plot pressure evolution
plt.subplot(4, 1, 3)
plt.plot(time, pressures, label='Pressure')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure Evolution over Time')
plt.legend()

# Plot equilibrium pressure evolution
plt.subplot(4, 1, 4)
plt.plot(time, equilibrium_pressures, label='Equilibrium Pressure')
plt.xlabel('Time (s)')
plt.ylabel('Equilibrium Pressure (Pa)')
plt.title('Equilibrium Pressure Evolution over Time')
plt.legend()

plt.tight_layout()
plt.show()

# Visualize the tank in 3D using pyvista
cylinder = pv.Cylinder(radius=tank_radius, height=tank_height, direction=(0, 0, 1))

# Plot the cylinder
plotter = pv.Plotter()
plotter.add_mesh(cylinder, color='blue', opacity=0.6)
plotter.add_axes()
plotter.show_grid()
plotter.show(title='3D Visualization of Hydrogen Storage Tank')

