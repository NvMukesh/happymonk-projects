Activation Functions:

ReLU (Rectified Linear Unit) function is used for the hidden layer activation: def relu(x).
Softmax function is used for the output layer activation: def softmax(x).
Layers:

Input Layer: The input layer size is determined by the number of features in the MNIST dataset, which is 784 (28x28 pixels).
Hidden Layer: The hidden layer has 256 neurons.
Output Layer: The output layer has 10 neurons, corresponding to the 10 classes in the MNIST dataset (digits 0-9).
Weight Initialization:

The weights (W1 and W2) are initialized using random values drawn from a normal distribution with a standard deviation of 0.01.
The biases (b1 and b2) are initialized as arrays of zeros.
Training:

The model is trained using mini-batch gradient descent with a specified batch size of 64.
The learning rate is set to 0.01.
The number of epochs is set to 10.
Loss Function:

The model uses categorical cross-entropy loss as the optimization objective.
Model Summary:

Input Layer: 784 neurons
Hidden Layer: 256 neurons with ReLU activation
Output Layer: 10 neurons with Softmax activation
