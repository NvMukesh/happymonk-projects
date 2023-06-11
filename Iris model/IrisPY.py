import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

# Load the IRIS dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode the target labels
encoder = OneHotEncoder(sparse=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))


# Set the initial parameters
np.random.seed(42)
m = X_train.shape[1]
n = 8
p = 4
k = len(np.unique(y_train))
epochs = 1000
learning_rate = 0.01

# Initialize the weights and biases
W1 = np.random.randn(m, n)
b1 = np.zeros((1, n))
W2 = np.random.randn(n, p)
b2 = np.zeros((1, p))
W3 = np.random.randn(p, k)
b3 = np.zeros((1, k))
K = np.random.randn(3)

# Define the activation function
def activation_function(x, k0, k1):
    return k0 + k1 * x

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2, K):
    z1 = np.dot(X, W1) + b1
    a1 = activation_function(z1, K[0], K[1])
    z2 = np.dot(a1, W2) + b2
    a2 = activation_function(z2, K[0], K[1])
    z3 = np.dot(a2, W3) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

# Softmax activation function
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Backward propagation
def backward_propagation(X, y, z1, a1, z2, a2, z3, a3, W2, K):
    m = X.shape[0]
    dz3 = a3 - y
    dw3 = (1 / m) * np.dot(a2.T, dz3)
    db3 = (1 / m) * np.sum(dz3, axis=0)
    da2 = np.dot(dz3, W3.T)
    dz2 = activation_derivative(a2, K[1]) * da2
    dw2 = (1 / m) * np.dot(a1.T, dz2)
    db2 = (1 / m) * np.sum(dz2, axis=0)
    da1 = np.dot(dz2, W2.T)
    dz1 = activation_derivative(a1, K[1]) * da1
    dw1 = (1 / m) * np.dot(X.T, dz1)
    db1 = (1 / m) * np.sum(dz1, axis=0)
    dK2 = np.array([np.mean(da2), np.mean(da2 * z2), np.mean(da2 * z2**2)])
    dK1 = np.array([np.mean(da1), np.mean(da1 * z1), np.mean(da1 * z1**2)])
    dK = dK2 + dK1
    return dw1, db1, dw2, db2, dw3, db3, dK


# Activation derivative
def activation_derivative(x, k1):
    return k1

# Training loop
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
f1_scores = []

for epoch in range(epochs):
    # Forward propagation
    z1, a1, z2, a2, z3, a3 = forward_propagation(X_train, W1, b1, W2, b2, K)

    # Backward propagation
    dw1, db1, dw2, db2, dw3, db3, dK = backward_propagation(X_train, y_train_encoded, z1, a1, z2, a2, z3, a3, W2, K)

    # Update parameters
    W1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dw3
    b3 -= learning_rate * db3
    K -= learning_rate * dK

    # Calculate train and test loss
    _, _, _, _, _, train_predictions = forward_propagation(X_train, W1, b1, W2, b2, K)
    _, _, _, _, _, test_predictions = forward_propagation(X_test, W1, b1, W2, b2, K)
    train_loss = -np.mean(np.sum(y_train_encoded * np.log(train_predictions), axis=1))
    test_loss = -np.mean(np.sum(y_test_encoded * np.log(test_predictions), axis=1))
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    # Calculate train and test accuracy
    train_accuracy = accuracy_score(y_train, np.argmax(train_predictions, axis=1))
    test_accuracy = accuracy_score(y_test, np.argmax(test_predictions, axis=1))
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # Calculate F1-Score
    f1 = f1_score(y_test, np.argmax(test_predictions, axis=1), average='weighted')
    f1_scores.append(f1)
    
    
for epoch in range(epochs):
    # Forward propagation
    z1, a1, z2, a2, z3, a3 = forward_propagation(X_train, W1, b1, W2, b2, K)

    # Backward propagation
    dw1, db1, dw2, db2, dw3, db3, dK = backward_propagation(X_train, y_train_encoded, z1, a1, z2, a2, z3, a3, W2, K)

    # Update parameters
    W1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dw3
    b3 -= learning_rate * db3
    K -= learning_rate * dK

    # Calculate train and test loss
    _, _, _, _, _, train_predictions = forward_propagation(X_train, W1, b1, W2, b2, K)
    _, _, _, _, _, test_predictions = forward_propagation(X_test, W1, b1, W2, b2, K)
    train_loss = -np.mean(np.sum(y_train_encoded * np.log(train_predictions), axis=1))
    test_loss = -np.mean(np.sum(y_test_encoded * np.log(test_predictions), axis=1))
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    # Calculate train and test accuracy
    train_accuracy = accuracy_score(y_train, np.argmax(train_predictions, axis=1))
    test_accuracy = accuracy_score(y_test, np.argmax(test_predictions, axis=1))
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # Calculate F1-Score
    f1 = f1_score(y_test, np.argmax(test_predictions, axis=1), average='weighted')
    f1_scores.append(f1)
    
    # Model summary
print("\nModel Summary:")
print("Number of features:", m)
print("Number of hidden nodes:", n)
print("Number of output nodes:", k)
print("Number of training examples:", X_train.shape[0])
print("Number of test examples:", X_test.shape[0])
print("Number of epochs:", epochs)


# Plot the loss function vs. epochs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_losses, label='Train')
plt.plot(range(epochs), test_losses, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy vs. epochs
plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_accuracies, label='Train')
plt.plot(range(epochs), test_accuracies, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Show the plots
plt.show()

# Print initial and final weights
print("Initial Weights:")
print("W1:\n", W1)
print("b1:\n", b1)
print("W2:\n", W2)
print("b2:\n", b2)
print("W3:\n", W3)
print("b3:\n", b3)
print("K:\n", K)
print("\nFinal Weights:")
print("W1:\n", W1)
print("b1:\n", b1)
print("W2:\n", W2)
print("b2:\n", b2)
print("W3:\n", W3)
print("b3:\n", b3)
print("K:\n", K)