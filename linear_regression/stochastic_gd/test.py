import numpy as np
import matplotlib.pyplot as plt

# Toy dataset
X = np.array([[1, 1], [1, 2], [1, 3]])  # bias term included
y = np.array([1, 2, 3])

theta = np.zeros(X.shape[1])
alpha = 0.1
iterations = 50

cost_history = []

def compute_cost(X, y, theta):
    m = len(y)
    return (1/(2*m)) * np.sum((X @ theta - y)**2)

for _ in range(iterations):
    for i in range(len(y)):
        error = (X[i] @ theta) - y[i]
        theta -= alpha * error * X[i]
    cost_history.append(compute_cost(X, y, theta))

print("SGD Theta:", theta)

# Plot cost function trajectory
plt.plot(range(iterations), cost_history, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("SGD Convergence Trajectory")
plt.show()
