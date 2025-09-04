#import required libraries 
import numpy as np
import matplotlib.pyplot as plt 


##Example data set : house size vs price
#X = feature (size in 1000 sq ft)
#Y = target (price in $100K)

X = np.array([3, 5, 7, 9, 11, 13, 15]) #sizes
Y = np.array([4, 8, 10, 14, 16, 20, 22]) #prices

X = X.reshape(-1,1) #converts this into a column vector from 1D array

X_b = np.c_[np.ones((X.shape[0], 1)), X] # Add a column of ones for theta0 (bias term), combine theta0 with other features
#The bias term (theta0) is the intercept of the line. It allows the line to shift up or down so it can fit the data better.

# Theta = parameters (theta0 and theta1), start with zeros
Theta = np.zeros((2, 1))  # shape (2,1) Size is 2 bacuse theta0 , theta1 = size

def hypothesis(X, Theta):
    return X.dot(Theta)

# Test initial hypothesis
print(hypothesis(X_b, Theta))

# Cost fucntion J(Theta)
def compute_cost(X, Y, Theta):
    m = len(Y)  # number of training examples
    predictions = hypothesis(X, Theta)
    error = predictions - Y.reshape(-1, 1)  # make Y a column vector
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost

# Test cost function with initial Theta
initial_cost = compute_cost(X_b, Y, Theta)
print("Initial cost:", initial_cost)

# Implement Gradient Descent
def gradient_descent(X, Y, Theta, alpha, iterations):
    m = len(Y)
    Y = Y.reshape(-1,1)  # ensure Y is a column vector
    cost_history = []

    for i in range(iterations):
        predictions = hypothesis(X, Theta)
        error = predictions - Y
        # Update Theta
        Theta = Theta - (alpha / m) * X.T.dot(error)
        # Compute and store cost
        cost = compute_cost(X, Y, Theta)
        cost_history.append(cost)
    
    return Theta, cost_history


# Run gradient Descent

alpha = 0.01       # learning rate
iterations = 1000  # number of steps

Theta_final, cost_history = gradient_descent(X_b, Y, Theta, alpha, iterations)

print("Final Theta:", Theta_final)
print("Final Cost:", cost_history[-1])


#Plot Cost Decrease
plt.plot(range(iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost J(Theta)")
plt.title("Cost Function Convergence")
plt.show()


# Plot training data
plt.scatter(X, Y, color='blue', label='Data points')

# Predicted values using final Theta
Y_pred = hypothesis(X_b, Theta_final)

# Plot regression line
plt.plot(X, Y_pred, color='red', label='Best fit line')

plt.xlabel("House Size (1000 sq ft)")
plt.ylabel("Price ($100k)")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

# Predict price for a new house of size 10
new_size = 100
new_size_b = np.array([1, new_size])  # include bias
predicted_price = new_size_b.dot(Theta_final)
print("Predicted price for size 10:", predicted_price)