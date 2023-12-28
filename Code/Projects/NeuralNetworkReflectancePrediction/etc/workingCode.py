import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ... [Other functions such as sigmoid, tanh, init_params, etc.] ...

# Mean Squared Error Loss Function
def mse_loss(A2, Y):
    return ((A2 - Y) ** 2).mean()

# Forward propagation with linear activation for the output layer
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = Z2  # Linear activation for the output layer
    return Z1, A1, Z2, A2

# Gradient descent with loss history for training and testing datasets
def gradient_descent(X_train, Y_train, X_test, Y_test, alpha, epochs):
    W1, b1, W2, b2 = init_params()
    m = X_train.shape[1]
    train_loss_history = []
    test_loss_history = []
    
    for i in range(epochs):
        # Shuffle the training data
        permutation = np.random.permutation(X_train.shape[1])
        X_train_shuffled = X_train[:, permutation]
        Y_train_shuffled = Y_train[:, permutation]

        # Forward and backward propagation
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train_shuffled)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X_train_shuffled, Y_train_shuffled, m)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # Calculate loss for plotting
        train_loss = mse_loss(A2, Y_train_shuffled)
        train_loss_history.append(train_loss)
        _, _, _, A2_test = forward_prop(W1, b1, W2, b2, X_test)
        test_loss = mse_loss(A2_test, Y_test)
        test_loss_history.append(test_loss)

        if i % 10 == 0:
            print(f"Epoch: {i}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return W1, b1, W2, b2, train_loss_history, test_loss_history

# Plot loss history for training and testing datasets
def plot_loss_history(train_loss_history, test_loss_history):
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(test_loss_history, label='Testing Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Training and Testing Loss History')
    plt.legend()
    plt.show()

# ... [Other parts of your code] ...

# Main execution
if __name__ == "__main__":
    # ... [Data loading and preprocessing] ...

    # Split data into features and target
    X = data[:, 1:].T  # Features (exclude the first column if it's an index or non-feature data)
    Y = data[:, 0:1].T  # Target is the first column (red component)

    # Split data into training and testing sets
    split_index = int(0.8 * X.shape[1])  # For an 80-20 split
    X_train, X_test = X[:, :split_index], X[:, split_index:]
    Y_train, Y_test = Y[:, :split_index], Y[:, split_index:]

    # Training parameters
    alpha = 0.1
    epochs = 500

    # Train the model
    W1, b1, W2, b2, train_loss_history, test_loss_history = gradient_descent(X_train, Y_train, X_test, Y_test, alpha, epochs)
    
    # Plot the loss history
    plot_loss_history(train_loss_history, test_loss_history)
    
    # Save the model weights
    save_weights(W1, b1, W2, b2)
