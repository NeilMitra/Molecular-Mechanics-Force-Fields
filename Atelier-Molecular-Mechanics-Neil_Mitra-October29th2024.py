import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
kb = 1.0  # Force constant for bond stretching
ktheta = 1.0  # Force constant for angle bending
k_torsion = 1.0  # Force constant for torsional
epsilon = 1.0  # Depth of van der Waals potential well
r0 = 1.0  # Equilibrium distance for van der Waals
q1, q2 = 1.0, -1.0  # Example charges for electrostatic interaction

# Generate data points
r = np.linspace(0.5, 2.0, 1000)  # Distance range
theta = np.linspace(0, 2*np.pi, 1000)  # Angle range
phi = np.linspace(0, 2*np.pi, 1000)  # Torsion angle range

# Energy functions
def bond_stretching(r, b0=1.0):
    return kb * (r - b0)**2

def angle_bending(theta, theta0=np.pi/2):
    return ktheta * (theta - theta0)**2

def torsional(phi):
    return k_torsion * (1 + np.cos(3*phi))  # Using n=3 as example

def electrostatic(r):
    return (q1 * q2) / r

def van_der_waals(r):
    return epsilon * ((r0/r)**12 - 2*(r0/r)**6)

# Create subplots
plt.figure(figsize=(15, 10))

# 1. Bond stretching
plt.subplot(2, 3, 1)
b0 = 1.0
r_bond = np.linspace(0.5, 1.5, 1000)
plt.plot(r_bond, bond_stretching(r_bond, b0))
plt.title('Bond Length Stretching')
plt.xlabel('Bond length (b)')
plt.ylabel('Energy')
plt.axvline(x=b0, color='k', linestyle='--', alpha=0.3)

# 2. Angle bending
plt.subplot(2, 3, 2)
theta0 = np.pi/2
theta_range = np.linspace(0, np.pi, 1000)
plt.plot(np.degrees(theta_range), angle_bending(theta_range, theta0))
plt.title('Bond Angle Bending')
plt.xlabel('Angle (θ) [degrees]')
plt.ylabel('Energy')
plt.axvline(x=np.degrees(theta0), color='k', linestyle='--', alpha=0.3)

# 3. Torsional
plt.subplot(2, 3, 3)
phi_range = np.linspace(0, 2*np.pi, 1000)
plt.plot(np.degrees(phi_range), torsional(phi_range))
plt.title('Torsional Angle Twisting')
plt.xlabel('Torsion angle (φ) [degrees]')
plt.ylabel('Energy')

# 4. Electrostatic
plt.subplot(2, 3, 4)
r_elec = np.linspace(0.5, 2.0, 1000)
plt.plot(r_elec, electrostatic(r_elec))
plt.title('Electrostatic Interaction')
plt.xlabel('Separation (r)')
plt.ylabel('Energy')

# 5. van der Waals
plt.subplot(2, 3, 5)
r_vdw = np.linspace(0.8, 2.0, 1000)
plt.plot(r_vdw, van_der_waals(r_vdw))
plt.title('van der Waals Interaction')
plt.xlabel('Separation (r)')
plt.ylabel('Energy')
plt.axvline(x=r0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()

# 3D Combined Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create mesh grid for 3D plot
X, Y = np.meshgrid(np.linspace(0.8, 2.0, 100), np.linspace(0.8, 2.0, 100))
Z = np.zeros_like(X)

# Combine van der Waals and electrostatic interactions for demonstration
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j] = van_der_waals(X[i,j]) + electrostatic(Y[i,j])

surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('Distance r1')
ax.set_ylabel('Distance r2')
ax.set_zlabel('Energy')
ax.set_title('Combined Interaction Energy')
fig.colorbar(surf)

plt.show()