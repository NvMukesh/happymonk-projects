# happymonk-projects
This repository contain ANN models with 1 hidden layer for MNIST, Bank-note, iris and Breast cancer dataset

Summary report on the Ann models



MNIST model
Activation Functions:

ReLU (Rectified Linear Unit) function is used for the hidden layer activation: def relu(x). Softmax function is used for the output layer activation: def softmax(x). Layers:

Input Layer: The input layer size is determined by the number of features in the MNIST dataset, which is 784 (28x28 pixels). Hidden Layer: The hidden layer has 256 neurons. Output Layer: The output layer has 10 neurons, corresponding to the 10 classes in the MNIST dataset (digits 0-9). Weight Initialization:

The weights (W1 and W2) are initialized using random values drawn from a normal distribution with a standard deviation of 0.01. The biases (b1 and b2) are initialized as arrays of zeros. Training:

The model is trained using mini-batch gradient descent with a specified batch size of 64. The learning rate is set to 0.01. The number of epochs is set to 10. Loss Function:

The model uses categorical cross-entropy loss as the optimization objective. Model Summary:

Input Layer: 784 neurons Hidden Layer: 256 neurons with ReLU activation Output Layer: 10 neurons with Softmax activation

F1-Score: 0.9214446223802867
Accuracy: 0.91

furthur detatils are in the Jupyter notebook in the folder of the repo



Bank-Note implementation:

The model used in the code is a Multi-Layer Perceptron (MLP) for binary classification. Here is a breakdown of the different aspects of the model:

Dataset:

The dataset used is the Banknote Authentication dataset. It contains features extracted from banknote images, such as variance, skewness, kurtosis, and entropy, to determine whether a banknote is authentic or not.
Activation Function:

The model uses a customized activation function called activation_function. This activation function is defined as k0 + k1 * x, where k0 and k1 are parameters that are updated during the training process. The activation function introduces non-linearity to the model, allowing it to learn complex relationships between the input features and the target labels.
Number of Neurons:

The model architecture consists of three layers: an input layer, a hidden layer, and an output layer.
The input layer has a dimension of m, which corresponds to the number of features in the dataset.
The hidden layer has n neurons. In the code, n is set to 8.
The output layer has k neurons, where k is the number of classes in the target variable. In the Banknote Authentication dataset, there are two classes: authentic and counterfeit banknotes.
Training and Evaluation:

The dataset is split into training and test sets using a test size of 0.2 (20% of the data is used for testing).
One-hot encoding is applied to the target labels using the OneHotEncoder from sklearn.preprocessing.
The model is trained using forward propagation and backward propagation. The weights and biases are updated iteratively using gradient descent optimization.
The training process is performed for a specified number of epochs (set to 1000 in the code) with a learning rate of 0.01.
During training, the loss (categorical cross-entropy) and accuracy are calculated for both the training and test sets. The loss and accuracy values are stored for each epoch.
Additionally, the F1-score is calculated to evaluate the model's performance on the test set.
Results and Visualization:

The final accuracy, F1-score, and confusion matrix on the test set are calculated and printed.
The loss function and accuracy are plotted against the number of epochs to visualize the training progress.
The initial and final weights and biases of the model are printed for each layer.

Model Summary:
Number of features: 4
Number of hidden nodes: 8
Number of output nodes: 2
Number of training examples: 1096
Number of test examples: 275
Number of epochs: 1000


Iris model:

Number of features (input nodes): m is determined by the shape of the input data X_train, representing the number of features in the IRIS dataset.

Number of hidden nodes: n is set to 8 in your code.

Number of output nodes: k represents the number of classes in the target variable. In this case, since IRIS dataset has three classes (setosa, versicolor, and virginica), k is set to 3.

Number of training examples: X_train.shape[0] represents the number of training examples.

Number of test examples: X_test.shape[0] represents the number of test examples.

Number of epochs: epochs is set to 1000, representing the number of times the model will iterate over the entire dataset during training.

The activation function used in the model is defined by the activation_function function. It takes input x and two parameters k0 and k1. The output is computed as k0 + k1 * x. The derivative of this activation function is a constant and is calculated by the activation_derivative function, which simply returns k1.

The model uses forward propagation and backward propagation to train the neural network. The forward_propagation function computes the output of each layer given the input and current weights. The backward_propagation function calculates the gradients of the weights and biases using backpropagation.

The softmax activation function is used for the output layer to obtain class probabilities. It is defined by the softmax function, which exponentiates the inputs and normalizes them to sum to 1.

The model is trained using the training loop, where the parameters are updated based on the gradients computed during backward propagation. The train and test loss, accuracy, and F1-score are calculated during each epoch and stored in the respective lists (train_losses, test_losses, train_accuracies, test_accuracies, f1_scores).

Test Accuracy: 0.40350877192982454
F1-Score: 0.9640287769784172

furthur detatils are in the Jupyter notebook in the folder of the repo

Breast cancer:
The implemented model is a neural network designed for binary classification using breast cancer data. It consists of an input layer, a hidden layer with 5 neurons, and an output layer with 2 neurons representing the two classes (0 or 1) in the dataset. The activation function used in the hidden layer is a customized activation function called "Ada-Act," and the output layer uses the softmax activation function. The model is trained using categorical cross-entropy loss and optimized using backpropagation with gradient descent.

Activation Function:
The Ada-Act activation function is defined as:
def ada_act(x, k0, k1):
    return k0 + k1 * x
It takes an input x and two parameters k0 and k1. The activation function performs an affine transformation of the input by scaling it with k1 and shifting it by k0. This activation function allows the neural network to learn different slopes and intercepts for different regions of the input space.

Number of Neurons:
The neural network has 5 neurons in the hidden layer. The choice of the number of neurons in a hidden layer is a design decision and can vary depending on the complexity of the problem and the size of the dataset. In this case, 5 neurons were chosen for the hidden layer.

The input layer has a dimension based on the number of features in the breast cancer dataset, and the output layer has 2 neurons to represent the binary classification problem.

Overall, the model architecture and the choice of activation functions and number of neurons are specific to the problem at hand and can be adjusted based on experimentation and performance evaluation.

Test Accuracy: 0.40350877192982454
F1-Score: 0.9640287769784172


furthur detatils are in the Jupyter notebook in the folder of the repo


In summary this MNIST model has high accuracy due the high size datset so the neuron trains itself better and avoids over fitting and updates very will and vice versa IRIS model accuracy reflect due the size of dataset
