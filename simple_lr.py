import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('studytime.csv')

# Normalize x and y
x_mean = data.x.mean()
x_std = data.x.std()
y_mean = data.y.mean()
y_std = data.y.std()

data['x_norm'] = (data.x - x_mean) / x_std
data['y_norm'] = (data.y - y_mean) / y_std

# Loss function for normalized data
def loss_function(m, b, points):
    n = len(points)
    error = 0
    for i in range(n):
        x = points.iloc[i].x_norm
        y = points.iloc[i].y_norm
        point_error = (y - (m * x + b)) ** 2
        error += point_error
    error = error / n
    return error

# Gradient descent for normalized data
def gradient_descent(m_curr, b_curr, points, alpha):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].x_norm
        y = points.iloc[i].y_norm
        m_gradient += -(2 / n) * x * (y - (m_curr * x + b_curr))
        b_gradient += -(2 / n) * (y - (m_curr * x + b_curr))
    m = m_curr - alpha * m_gradient
    b = b_curr - alpha * b_gradient
    return m, b

# Initialize parameters
m = 0
b = 0
learning_rate = 0.01
epochs = 1000

# Train on normalized data
for i in range(epochs):
    m, b = gradient_descent(m, b, data, learning_rate)
    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss_function(m, b, data)}")

# Convert normalized m and b back to original scale
m_orig = m * (y_std / x_std)
b_orig = y_mean - m_orig * x_mean + b * y_std

print("Gradient Descent (original scale):", m_orig, b_orig)

# Plot
plt.scatter(data.x, data.y, color="black")
x_vals = np.linspace(min(data.x), max(data.x), 300)
y_vals = m_orig * x_vals + b_orig
plt.plot(x_vals, y_vals, color='red', label='Gradient Descent')
plt.show()
