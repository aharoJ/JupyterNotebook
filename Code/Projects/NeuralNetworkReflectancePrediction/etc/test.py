import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys


# Sigmoid and its derivative
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s * (1 - s)

# Tanh and its derivative
def tanh(Z):
    return np.tanh(Z)

def tanh_derivative(Z):
    return 1 - np.tanh(Z)**2

# Initialization of parameters
def init_params():
    W1 = np.random.rand(10, 3) - 0.5  # Adjusted to match the input size (3 features for RGB)
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(1, 10) - 0.5  # Output layer stays the same
    b2 = np.random.rand(1, 1) - 0.5
    return W1, b1, W2, b2


# ReLU and its derivative
def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

# Softmax function
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Forward propagation for binary classification
def forward_prop_binary(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# One hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Backward propagation
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Update parameters
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

# Prediction
def get_predictions(A2):
    return np.argmax(A2, 0)

# Accuracy calculation
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Gradient descent for binary classification
def gradient_descent_binary(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    m = X.shape[1]
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop_binary(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0: # every nth we iterate a s.o.u.t
            predictions = get_predictions(A2)
            print("Iteration: ", i)
            print(get_accuracy(predictions, Y))
            sys.stdout.flush()
    return W1, b1, W2, b2

# Load and label data
def load_and_label_data(file_path, red_threshold=0.6):
    data = np.loadtxt(file_path)
    labels = (data[:, 0] > red_threshold).astype(int)
    return data, labels

# Plot loss history
def plot_loss_history(loss_history):
    plt.plot(loss_history)
    plt.ylabel('Loss')
    plt.xlabel('Iteration (in tens)')
    plt.title('Loss History')
    plt.show()

# Save weights
def save_weights(W1, b1, W2, b2, filename='weights.txt'):
    with open(filename, 'w') as f:
        np.savetxt(f, W1)
        f.write("\n")
        np.savetxt(f, b1)
        f.write("\n")
        np.savetxt(f, W2)
        f.write("\n")
        np.savetxt(f, b2)

# Main execution
if __name__ == "__main__":
    file_path = "red-fabric.txt"
    data, labels = load_and_label_data(file_path)

    # Normalize data and split (assuming data needs normalization, adjust as needed)
    data = data / data.max(axis=0)
    split = int(0.8 * len(data))
    X_train, X_val = data[:split].T, data[split:].T
    Y_train, Y_val = labels[:split], labels[split:]

    # Training parameters
    alpha = 0.1
    iterations = 500

    # Train the model
    W1, b1, W2, b2 = gradient_descent_binary(X_train, Y_train, alpha, iterations)

    # Save the model weights
    save_weights(W1, b1, W2, b2)
