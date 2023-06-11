import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X1 = mnist.data.astype('float32') / 255.0
X = X1.values
y1= mnist.target.astype('int')
y = y1.values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert y_train and y_test to NumPy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

# Define the activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Define the number of neurons in each layer
input_dim = X_train.shape[1]
hidden_dim = 256
output_dim = y_train_encoded.shape[1]

# Initialize the weights and biases
W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros(output_dim)

# Set hyperparameters
learning_rate = 0.01
num_epochs = 10
batch_size = 64
num_batches = X_train.shape[0] // batch_size

# Training loop
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    # Shuffle the training data
    indices = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train_encoded[indices]

    epoch_loss = 0

    # Mini-batch training
    for batch in range(num_batches):
        start = batch * batch_size
        end = start + batch_size

        # Forward propagation
        h = relu(np.dot(X_train_shuffled[start:end], W1) + b1)
        y_pred = softmax(np.dot(h, W2) + b2)

        # Compute the categorical cross-entropy loss
        loss = -np.mean(y_train_shuffled[start:end] * np.log(y_pred + 1e-8))
        epoch_loss += loss

        # Backpropagation
        grad_y_pred = (y_pred - y_train_shuffled[start:end]) / batch_size
        grad_W2 = np.dot(h.T, grad_y_pred)
        grad_b2 = np.sum(grad_y_pred, axis=0)
        grad_h = np.dot(grad_y_pred, W2.T)
        grad_relu = grad_h * (h > 0)
        grad_W1 = np.dot(X_train_shuffled[start:end].T, grad_relu)
        grad_b1 = np.sum(grad_relu, axis=0)

        # Update parameters
        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1
        W2 -= learning_rate * grad_W2
        b2 -= learning_rate * grad_b2

    # Compute the average loss for the epoch
    epoch_loss /= num_batches
    train_losses.append(epoch_loss)

    # Evaluate on the test set
    h = relu(np.dot(X_test, W1) + b1)
    y_pred = softmax(np.dot(h, W2) + b2)
    test_loss = -np.mean(y_test_encoded * np.log(y_pred + 1e-8))
    test_losses.append(test_loss)

    # Compute accuracies
    y_train_pred = np.argmax(np.dot(relu(np.dot(X_train, W1) + b1), W2) + b2, axis=1)
    y_test_pred = np.argmax(y_pred, axis=1)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # Print the loss and accuracy for the epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, "
          f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plot the loss function vs. epochs
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Print the final parameter values
print("\nFinal Parameter Values:")
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)

# Plot the initial and final weight comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(W1.flatten(), bins=50, alpha=0.5, color='blue')
plt.title("Initial Weight Distribution")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.subplot(1, 2, 2)
plt.hist(W2.flatten(), bins=50, alpha=0.5, color='red')
plt.title("Final Weight Distribution")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Plot the accuracy vs. epoch
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Print the F1-Score
test_f1_score = f1_score(y_test, y_test_pred, average='macro')
print("\nTest F1-Score:", test_f1_score)

# Make predictions on the test set
predictions = np.argmax(y_pred, axis=1)
print("\nSample Predictions:")
for i in range(10):
    print("Predicted:", predictions[i], "Actual:", y_test[i])