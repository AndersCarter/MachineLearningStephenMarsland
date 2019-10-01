import numpy as np

class MLP:

    """ Represents a Multi-layered Perceptron with a single hidden layers"""

    def __init__(self, i_count, h_count, o_count, **kwargs):

        """
        Initializes a new MPL Object

        Arguments
        i_count | Integer : The number of neurons in the input layer
        h_count | Integer : The number of neurons in the hidden layer
        o_count | Integer : The number of neurons in the output layer

        Kwargs
        beta            | Float  : Beta value to use in sigmoid activation functions
        momentum        | Float  : How much momentum to apply to the weight changes as they are updated
        activation_type | String : The output layer activation function type
        eta             | Float  : The Learning Rate of the Network
        """

        ## Initialize Constants
        self.beta = kwargs["beta"] if "beta" in kwargs else 1.0
        self.momentum = kwargs["momentum"] if "momentum" in kwargs else 0.9
        self.activation_type = kwargs["activation_type"] if "activation_type" in kwargs else "logistic"
        self.eta = kwargs["eta"] if "eta" in kwargs else 0.25

        ## Intitalize Weights
        self.input_weights = (np.random.rand(i_count + 1, h_count) - 0.5) * 2 / np.sqrt(i_count)
        self.hidden_weights = (np.random.rand(h_count + 1, o_count) - 0.5) * 2 / np.sqrt(h_count)

        ## Working Hidden Layer
        self.hidden = None

    def eval(self, input):

        """ Evaluates a single input to the network and returns the output """

        ## Add bias to input
        input = np.array(input) if type(input) != np.array else input
        input = np.concatenate((input, [-1]))
        input = input.reshape((1, input.size))
        return 1 if self.forward(input)[0,0] > 0.5 else 0


    def forward(self, inputs):
        """
        Moves the network forward on the given input.

        Arguments
        inputs | Numpy Array : The Inputs to evaluate

        """

        def softmax(x):
            normalisers = np.sum(np.exp(x), axis = 1) * np.ones((1, x.shape[0]))
            return np.transpose(np.transpose(np.exp(x)) / normaliser)

        ## Activation Functions
        activation_funcs = {
            "linear"   : lambda x: x,
            "logistic" : lambda x: 1.0 / (1.0 + np.exp(-self.beta * x)),
            "softmax"  : softmax
        }

        ## Evaluate Hidden Neurons
        self.hidden = np.dot(inputs, self.input_weights)
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        ## Evaluate Output Neurons
        output = np.dot(self.hidden, self.hidden_weights)
        return activation_funcs[self.activation_type](output)


    def train(self, inputs, targets, max_iterations = 1000):

        """
        Trains the algorithm with the given inputs and targets

        Arguments
        inputs         | Numpy Array : A Matrix of all training inputs
        targets        | Numpy Array : An Array of all target values that correspond with the given inputs
        max_iterations | Integer     : The maximum number of iterations that the network will train
        """

        ## Add Bias Term to Inputs
        inputs = np.concatenate((inputs, -np.ones((inputs.shape[0], 1))), axis = 1)

        ## Initialize Update Arrays
        update_input = np.zeros(self.input_weights.shape)
        update_hidden = np.zeros(self.hidden_weights.shape)

        for i in range(max_iterations):

            ## Evaluate Inputs
            outputs = self.forward(inputs)

            ## Calculate Error
            error = 0.5 * np.sum((outputs - targets) ** 2)
            if i % 100 == 0:
                print(f"Iteration: {i} Error: {error}")

            ## Changes for different types of output nerurons
            neuron_derivatives = {
                "linear"   : lambda: (outputs - targets) / inputs.shape[0],
                "logistic" : lambda: self.beta * (outputs - targets) * outputs * (1.0 - outputs),
                "softmax"  : lambda: (outputs - targets) * (outputs * (-outputs) + outputs) / inputs.shape[0]
            }

            deltao = neuron_derivatives[self.activation_type]()
            deltah = self.hidden * self.beta * (1.0 - self.hidden) * np.dot(deltao, np.transpose(self.hidden_weights))

            update_input = self.eta * (np.dot(np.transpose(inputs), deltah[:,:-1])) + self.momentum * update_input
            update_hidden = self.eta * (np.dot(np.transpose(self.hidden), deltao)) + self.momentum * update_hidden

            self.input_weights -= update_input
            self.hidden_weights -= update_hidden


if __name__ == "__main__":

    np.random.seed(1234)

    ## Logical AND Network
    print("Logical AND")
    and_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_targets = np.array([0, 0, 0, 1]).reshape(4, 1)

    p = MLP(2, 3, 1)
    p.train(and_inputs, and_targets, 1001)

    print(f"Inputs: 0 0 Eval: {p.eval([0, 0])}")
    print(f"Inputs: 1 0 Eval: {p.eval([1, 0])}")
    print(f"Inputs: 0 1 Eval: {p.eval([0, 1])}")
    print(f"Inputs: 1 1 Eval: {p.eval([1, 1])}")
    print("")

    ## Logical XOR Network
    print("Logical XOR")
    xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_targets = np.array([0, 1, 1, 0]).reshape(4, 1)

    p = MLP(2, 3, 1)
    p.train(xor_inputs, xor_targets, 5001)

    print(f"Inputs: 0 0 Eval: {p.eval([0, 0])}")
    print(f"Inputs: 1 0 Eval: {p.eval([1, 0])}")
    print(f"Inputs: 0 1 Eval: {p.eval([0, 1])}")
    print(f"Inputs: 1 1 Eval: {p.eval([1, 1])}")
