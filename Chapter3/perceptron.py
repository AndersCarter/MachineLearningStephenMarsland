import numpy as np

## A Perceptron is a simple machine learning concept. The one below is designed
## to learn the Logical Or operation using the McCulloch and Pitts neuron model
class Perceptron:

    ## Initializes a new Perception Object
    def __init__(self):
        self.weights = None

    ## Trains the Perceptron
    def train(self, inputs, targets, eta, maxIterations):

        # Initializes the weights
        self.weights = np.random.rand(inputs.shape[1] + 1, targets.shape[1]) * 0.1 - 0.05

        # Add the bias to the inputs
        inputs = np.insert(inputs, 0, -1, axis = 1)

        for i in range(maxIterations):

            print(f"Iteration: {i}")
            print(self.weights)

            # Compute Activations
            activations = np.dot(inputs, self.weights)

            # Threshold Activations
            activations = np.where(activations > 0, 1, 0)
            print("Final outputs are")
            print(activations)
            print("")

            # Updage Weights
            self.weights -= eta * np.dot(np.transpose(inputs), activations - targets)

    ## Evals the Perceptron
    def eval(self, input):
        input = np.insert(input, 0, -1)
        return np.where(np.dot(np.transpose(self.weights), input) > 0, 1, 0)[0]
