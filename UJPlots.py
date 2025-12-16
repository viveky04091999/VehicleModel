import numpy as np
import matplotlib.pyplot as plt

# Constants
UUJ = float(input("What is the UUJ? (in degrees): "))
LUJ = float(input("What is the LUJ? (in degrees): "))

n = 1.4
omega0 = 1  # deg/s (for scaling angular velocities)

# theta0 range
theta0 = np.linspace(-360*n, 360*n, 1000)

# Initialize arrays
theta1 = []
theta2 = []

# Loop with quadrant correction
for t0 in theta0:
    factor1 = int(np.floor((t0 + 90) / 180))
    theta0_rad = np.radians(t0)
    
    raw1 = np.arctan(np.tan(theta0_rad) * np.cos(np.radians(UUJ)))
    theta1_val = factor1 * np.pi + raw1
    raw2 = np.arctan(np.tan(theta1_val) / np.cos(np.radians(LUJ)))
    factor2 = int(np.floor((np.degrees(theta1_val) + 90) / 180))
    theta2_val = factor2 * np.pi + raw2

    theta1.append(np.degrees(theta1_val))
    theta2.append(np.degrees(theta2_val))

theta1 = np.array(theta1)
theta2 = np.array(theta2)

# Compute delta angles
dtheta1 = theta1 - theta0
dtheta2 = theta2 - theta0

# First plot: dtheta1 and dtheta2
plt.figure(figsize=(10, 6))
plt.plot(theta0, dtheta1, label='dtheta1 = θ1 - θ0', color='blue')
plt.plot(theta0, dtheta2, label='dtheta2 = θ2 - θ0', color='green')
plt.xlabel('theta0 (deg)')
plt.ylabel('Angle Difference (deg)')
plt.title('dtheta1 and dtheta2 vs theta0')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Compute angular velocities
dtheta1_dtheta0 = np.gradient(theta1, theta0)
dtheta2_dtheta0 = np.gradient(theta2, theta0)
omega1 = dtheta1_dtheta0 * omega0
omega2 = dtheta2_dtheta0 * omega0

# Second plot: omega1 and omega2
plt.figure(figsize=(10, 6))
plt.plot(theta0, omega1, label='omega1 = dθ1/dθ0', color='blue')
plt.plot(theta0, omega2, label='omega2 = dθ2/dθ0', color='green')
plt.xlabel('theta0 (deg)')
plt.ylabel('Angular Velocity (normalized)')
plt.title('Angular Velocities omega1 and omega2 vs theta0')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
