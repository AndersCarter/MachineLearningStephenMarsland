import numpy as np

## A Perceptron is a simple machine learning concept. The one below is designed
## to learn the Logical Or operation using the McCulloch and Pitts neuron model
class Perceptron:

    ## Initializes a new Perception Object
    def __init__(self, i_count, o_count, **kwargs):

        """
            Arguments
            i_count | Integer : The number of input nodes
            o_count | Integer : The number of output nodes

            Keyword Arguments
            eta | Float : The learning rate
        """

        ## Initialize Weights
        self.weights = np.random.rand(i_count + 1, o_count) * 0.1 - 0.05
        self.eta = 0.25 if "eta" not in kwargs else kwargs["eta"]

    ## Trains the Perceptron
    def train(self, inputs, targets, maxIterations):

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
            self.weights -= self.eta * np.dot(np.transpose(inputs), activations - targets)

    ## Evals the Perceptron
    def eval(self, input):
        input = np.insert(input, 0, -1)
        return np.where(np.dot(np.transpose(self.weights), input) > 0, 1, 0)[0]
