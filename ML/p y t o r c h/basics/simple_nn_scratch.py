import numpy as np

# define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)

    def feedforward(self, inputs):
        self.hidden = sigmoid(np.dot(inputs, self.weights1))
        self.output = sigmoid(np.dot(self.hidden, self.weights2))
        return self.output

    def backpropagation(self, inputs, target_output, learning_rate):
        error = target_output - self.output
        delta_output = error * sigmoid_derivative(self.output)
        error_hidden = delta_output.dot(self.weights2.T)
        delta_hidden = error_hidden * sigmoid_derivative(self.hidden)
        self.weights2 += self.hidden.T.dot(delta_output) * learning_rate
        self.weights1 += inputs.T.dot(delta_hidden) * learning_rate

# create a neural network with 2 inputs, 4 hidden neurons, and 1 output
nn = NeuralNetwork(2, 4, 1)

# train the network with XOR dataset
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_outputs = np.array([[0], [1], [1], [0]])

for i in range(10000):
    nn.feedforward(inputs)
    # learning rate = 0.1
    nn.backpropagation(inputs, target_outputs, 0.1)

# test the network
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print(nn.feedforward(test_inputs))