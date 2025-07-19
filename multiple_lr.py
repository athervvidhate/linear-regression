import numpy as np
import pandas as pd
import os

csv_path = os.path.join(os.path.dirname(__file__), 'california_housing_train.csv')
df = pd.read_csv(csv_path)

# feature scaling function
def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # avoids division by zero
    std = np.where(std == 0, 1, std)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

def predict(X,w,b):
    return np.dot(X, w) + b


def calculate_error(X,y,w,b):
    m = X.shape[0]

    y_pred = predict(X,w,b)

    # MSE function
    sum = np.sum((y_pred - y) ** 2)
    residual = sum / (2 * m)
    return residual


def grad_step(X, y, w, b):
    # number of training examples
    m = X.shape[0]

    y_pred = predict(X,w,b)
    errors = y_pred - y

    # calculate average gradients
    partial_dw = np.dot(X.T, errors) / m
    partial_db = np.sum(errors) / m

    return partial_dw, partial_db


def gradient_descent(X, y, w, b, epochs=1000, alpha=0.01, log_interval=100):
    history = {}

    for epoch in range(epochs):
        partial_dw, partial_db = grad_step(X, y, w, b)

        w = w - alpha * partial_dw
        b = b - alpha * partial_db

        if epoch % log_interval == 0:
            cost = calculate_error(X,y,w,b)
            history[epoch] = cost

    return history, w, b

# Train/test split used for model testing
def shuffle_dataset(X, y):

    length = np.arange(X.shape[0])
    np.random.shuffle(length)

    return X[length], y[length]

def train_test_split(X, y, test_size=0.5, shuffle=True):
    if shuffle:
        X, y = shuffle_dataset(X, y)
    if test_size <1 :
        train_ratio = len(y) - int(len(y) *test_size)
        X_train, X_test = X[:train_ratio], X[train_ratio:]
        y_train, y_test = y[:train_ratio], y[train_ratio:]
        return X_train, X_test, y_train, y_test
    elif test_size in range(1,len(y)):
        X_train, X_test = X[test_size:], X[:test_size]
        y_train, y_test = y[test_size:], y[:test_size]
        return X_train, X_test, y_train, y_test
