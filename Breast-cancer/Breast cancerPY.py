import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load the Bank Note Authentication dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Ada-Act activation function
def ada_act(x, k0, k1):
    return k0 + k1 * x

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights and biases
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        
        self.k0 = None
        self.k1 = None
        self.final_k0 = None
        self.final_k1 = None
        
    def initialize_weights(self):
        self.w1 = np.random.randn(self.input_dim, self.hidden_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.w2 = np.random.randn(self.hidden_dim, self.output_dim)
        self.b2 = np.zeros((1, self.output_dim))
        
        self.k0 = np.random.randn()
        self.k1 = np.random.randn()
        
    def forward_propagation(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = ada_act(self.z1, self.k0, self.k1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = softmax(self.z2)
        
    def backward_propagation(self, X, y):
        m = X.shape[0]
        
        # Calculate gradients
        dz2 = self.a2 - y.reshape(-1, 1)
        dw2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0)
        da1 = np.dot(dz2, self.w2.T)
        dz1 = ada_act_derivative(self.z1, self.k0, self.k1) * da1
        dw1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0)
        
        # Update weights and biases
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        
    def train(self, X, y, epochs, learning_rate):
        self.learning_rate = learning_rate
        
        self.initialize_weights()
        
        self.loss_history = []
        self.accuracy_history = []
        
        for epoch in range(epochs):
            self.forward_propagation(X)
            self.backward_propagation(X, y)
            
            # Calculate training loss and accuracy
            loss = categorical_cross_entropy(y, self.a2)
            accuracy = accuracy_score(np.argmax(y, axis=1), np.argmax(self.a2, axis=1))
            
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            
        self.final_k0 = self.k0
        self.final_k1 = self.k1
        self.final_w1 = self.w1
        self.final_b1 = self.b1
        self.final_w2 = self.w2
        self.final_b2 = self.b2

    def calculate_accuracy(self, X, y):
        self.forward_propagation(X)
        predictions = np.argmax(self.a2, axis=1)
        return accuracy_score(np.argmax(y, axis=1), predictions)
    
    def predict(self, X):
        self.forward_propagation(X)
        return np.argmax(self.a2, axis=1)


# Softmax activation function
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Derivative of the Ada-Act activation function
def ada_act_derivative(x, k0, k1):
    return k1

# Categorical cross-entropy loss function
def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# Create and train the neural network
input_dim = X_train.shape[1]
output_dim = 2  # Number of classes (0 or 1)
nn = NeuralNetwork(input_dim, hidden_dim=5, output_dim=output_dim)
nn.train(X_train, y_train.reshape(-1, 1), epochs=100, learning_rate=0.01)

# Make predictions on the test set
predictions = nn.predict(X_test)

# Calculate accuracy
accuracy = nn.calculate_accuracy(X_test, y_test.reshape(-1, 1))
print("Test Accuracy:", accuracy)

# Calculate F1-Score
f1 = f1_score(y_test, predictions)
print("F1-Score:", f1)

# Calculate confusion matrix
confusion_mat = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(confusion_mat)

# Plot loss function vs. epochs
plt.plot(range(1, len(nn.loss_history) + 1), nn.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function vs. Epochs')
plt.show()

# Plot accuracy vs. epochs
plt.plot(range(1, len(nn.accuracy_history) + 1), nn.accuracy_history)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epochs')
plt.show()

# Print initial and final weights
print("Initial Weights:")
print("k0:", nn.k0)
print("k1:", nn.k1)
print("w1:", nn.w1)
print("b1:", nn.b1)
print("w2:", nn.w2)
print("b2:", nn.b2)

print("\nFinal Weights:")
print("k0:", nn.final_k0)
print("k1:", nn.final_k1)
print("w1:", nn.final_w1)
print("b1:", nn.final_b1)
print("w2:", nn.final_w2)
print("b2:", nn.final_b2)