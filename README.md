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
T_ref = 298  # Reference temperature in K
activation_energy = 50000  # Activation energy in J/mol
rate_constant = 0.001  # Rate constant
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
initial_mass = 0
initial_fraction = 0  # Initial fraction of hydrogen absorbed

# Define the probe positions (for simplicity, we'll use a grid)
n_probes = 10
probe_positions = np.linspace(0, tank_height, n_probes)

# Initialize arrays to store results
temperatures = np.full((n_probes, len(time)), initial_temperature)
pressures = np.full((n_probes, len(time)), initial_pressure)
equilibrium_pressures = np.full((n_probes, len(time)), initial_pressure)
velocities_darcy = np.zeros((n_probes, len(time)))
velocities_turbulent = np.zeros((n_probes, len(time)))
velocities_laminar = np.zeros((n_probes, len(time)))
velocities_transition = np.zeros((n_probes, len(time)))
mass_stored_darcy = np.zeros(len(time))
mass_stored_turbulent = np.zeros(len(time))
mass_stored_laminar = np.zeros(len(time))
mass_stored_transition = np.zeros(len(time))
fractions = np.full((n_probes, len(time)), initial_fraction)

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
def hydrogen_absorption(X, t, T):
    return rate_constant * (1 - X) * np.exp(-activation_energy / (R * T))

# Simulation loop
for t in range(1, len(time)):
    for i in range(n_probes):
        # Calculate the pressure gradient (simplified)
        if i == 0:
            dP = pressures[i, t-1] - pressures[i+1, t-1]
        elif i == n_probes - 1:
            dP = pressures[i-1, t-1] - pressures[i, t-1]
        else:
            dP = (pressures[i-1, t-1] - pressures[i+1, t-1]) / 2

        # Calculate velocities based on flow regimes
        velocities_darcy[i, t] = darcy_flow(K, dP, mu, tank_height/n_probes)
        velocities_turbulent[i, t] = turbulent_flow(dP, rho, tank_height/n_probes)
        velocities_laminar[i, t] = laminar_flow(mu, dP, tank_height/n_probes, tank_radius)
        velocities_transition[i, t] = transition_flow(dP, rho, mu, tank_height/n_probes, tank_radius)

        # Update temperature (considering heat transfer in porous media)
        Q = delta_H * fractions[i, t-1]
        T_prev = temperatures[i, t-1]
        convective_term = velocities_darcy[i, t] * (T_amb - T_prev)
        conductive_term = k_eff * (T_amb - T_prev) / (tank_height/n_probes)
        temperatures[i, t] = T_prev + dt * (Q / (m_hydride * c_p) + convective_term + conductive_term)

        # Update hydrogen fraction absorbed using ODE solver
        fractions[i, t] = odeint(hydrogen_absorption, fractions[i, t-1], [0, dt], args=(temperatures[i, t],))[-1]

        # Update pressure using the ideal gas law
        pressures[i, t] = P_amb * (1 - fractions[i, t])

        # Calculate equilibrium pressure using Van't Hoff equation
        equilibrium_pressures[i, t] = np.exp(delta_H / R * (1 / T_ref - 1 / temperatures[i, t]))

    # Calculate mass stored
    mass_stored_darcy[t] = capacity * (tank_volume - np.sum(velocities_darcy[:, t] * dt))
    mass_stored_turbulent[t] = capacity * (tank_volume - np.sum(velocities_turbulent[:, t] * dt))
    mass_stored_laminar[t] = capacity * (tank_volume - np.sum(velocities_laminar[:, t] * dt))
    mass_stored_transition[t] = capacity * (tank_volume - np.sum(velocities_transition[:, t] * dt))

# Plot results
plt.figure(figsize=(12, 10))

# Plot mass stored comparison
plt.subplot(5, 1, 1)
plt.plot(time, mass_stored_darcy, label='Darcy Flow')
plt.plot(time, mass_stored_turbulent, label='Turbulent Flow')
plt.plot(time, mass_stored_laminar, label='Laminar Flow')
plt.plot(time, mass_stored_transition, label='Transition Flow')
plt.xlabel('Time (s)')
plt.ylabel('Mass Stored (mol)')
plt.title('Hydrogen Mass Stored over Time')
plt.legend()

# Plot temperature at different probe positions for one regime (Darcy Flow)
plt.subplot(5, 1, 2)
for i in range(n_probes):
    plt.plot(time, temperatures[i, :], label=f'Probe {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title('Temperature at Different Probe Positions')
plt.legend()

# Plot pressure at different probe positions for one regime (Darcy Flow)
plt.subplot(5, 1, 3)
for i in range(n_probes):
    plt.plot(time, pressures[i, :], label=f'Probe {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure at Different Probe Positions')
plt.legend()

# Plot equilibrium pressure at different probe positions for one regime (Darcy Flow)
plt.subplot(5, 1, 4)
for i in range(n_probes):
    plt.plot(time, equilibrium_pressures[i, :], label=f'Probe {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Equilibrium Pressure (Pa)')
plt.title('Equilibrium Pressure at Different Probe Positions')
plt.legend()

# Plot velocities for different flow regimes at the first probe position
plt.subplot(5, 1, 5)
plt.plot(time, velocities_darcy[0, :], label='Darcy Flow')
plt.plot(time, velocities_turbulent[0, :], label='Turbulent Flow')
plt.plot(time, velocities_laminar[0, :], label='Laminar Flow')
plt.plot(time, velocities_transition[0, :], label='Transition Flow')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocities at Probe 1 for Different Flow Regimes')
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

