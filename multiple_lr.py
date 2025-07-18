import numpy as np
import pandas as pd

df = pd.read_csv('california_housing_train.csv')


def predict(X,w,b):
    m = X.shape[0]

    y_pred = np.zeros(m)

    for i in range(m):
        y_pred[i] = np.dot(w, X[i]) + b
    
    return y_pred


def calculate_error(X,y,w,b):
    m = X.shape[0]

    y_pred = predict(X,w,b)

    # MSE function
    sum = np.sum((y_pred - y)**2)
    residual = sum / (2 * m)
    return residual


def grad_step(X, y, w, b):
    # Number of training examples
    m = X.shape[0]

    # Number of features
    n = X.shape[1]

    partial_dw = np.zeros(n)
    partial_db = 0

    for i in range(m):
        error = np.dot(w, X[i]) + b - y[i]

        partial_dw += error * X[i]
        partial_db += error

    partial_dw /= m    
    partial_db /= m

    return partial_dw, partial_db


def gradient_descent(X, y, w, b, epochs=1000, alpha=0.001, log_interval=100):
    history = {}

    for epoch in range(epochs):
        partial_dw, partial_db = grad_step(X,y,w,b)

        w = w - alpha*partial_dw
        b = b - alpha*partial_db

        if epoch % log_interval == 0:
            cost = calculate_error(X,y,w,b)
            history[epoch] = cost

            print(f"Iteration {epoch} - Cost: {cost}")
    return history, w, b

