import numpy as np
import matplotlib.pyplot as plt

# Toy dataset (bias term + feature)
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])

# Compute Normal Equation solution
theta_ne = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"Normal Equation Theta: θ0 = {theta_ne[0]:.3f}, θ1 = {theta_ne[1]:.3f}")

# Prepare cost surface
theta0_vals = np.linspace(-1, 3, 50)
theta1_vals = np.linspace(0, 2, 50)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i, t0 in enumerate(theta0_vals):
    for j, t1 in enumerate(theta1_vals):
        t = np.array([t0, t1])
        J_vals[i, j] = np.sum((X @ t - y)**2) / (2 * len(y))

# Find minimum cost
min_cost = np.min(J_vals)
print(f"Minimum Cost at Normal Equation solution: {min_cost:.6f}")

# Plot contour
plt.figure(figsize=(8,6))
contour = plt.contour(theta0_vals, theta1_vals, J_vals.T, levels=20, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
plt.plot(theta_ne[0], theta_ne[1], 'ro', label='Normal Equation Solution')
plt.xlabel('Theta 0 (bias)')
plt.ylabel('Theta 1 (slope)')
plt.title('Cost Function Contour & Normal Equation Solution')
plt.legend()
plt.grid(True)
plt.colorbar(contour, label='Cost')
plt.show()
