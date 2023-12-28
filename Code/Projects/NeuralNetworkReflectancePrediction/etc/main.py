import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
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

# Forward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
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

# Cross-Entropy Loss
def cross_entropy_loss(A2, Y):
    m = Y.shape[0]
    one_hot_Y = one_hot(Y)
    cost = -np.sum(one_hot_Y * np.log(A2)) / m
    return cost

# Gradient descent with loss and accuracy history
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    m = X.shape[1]  # Number of training examples
    loss_history = []  # To store the loss at each iteration
    accuracy_history = []  # To store the accuracy at each iteration
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        loss = cross_entropy_loss(A2, Y)
        loss_history.append(loss)  # Record the loss
        
        predictions = get_predictions(A2)
        accuracy = get_accuracy(predictions, Y)
        accuracy_history.append(accuracy)  # Record the accuracy
        
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 10 == 0:
            print(f"Iteration: {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
    return W1, b1, W2, b2, loss_history, accuracy_history

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

# Main Execution
if __name__ == "__main__":
    # Load and preprocess the data
    X_train, Y_train, X_test, Y_test = load_and_preprocess_data()

    # Set hyperparameters
    alpha = 0.01  # Learning rate
    epochs = 1000
    batch_size = 32  # For stochastic gradient descent

    # Train the model
    W1, b1, W2, b2, loss_history, accuracy_history = gradient_descent(
        X_train, Y_train, alpha, epochs, batch_size
    )

    # Evaluate the model
    test_loss, test_accuracy = evaluate_model(W1, b1, W2, b2, X_test, Y_test)

    # Plot training and testing errors
    plot_loss_history(loss_history, test_loss)

    # Save final weights
    save_weights(W1, b1, W2, b2)

    # Additional: Run with different learning rates for extra credit
