import numpy as np

class MPL:

    """ A Multilayered Perceptron with a single hidden layer """

    def __init__(self, ineuron_count, hneuron_count, oneuron_count, eta, momentum, max_iterations):

        """
        Initializaes a new Multilayered Perceptions

        ineuron_count  - Integer - The number of input neurons
        hneuron_count  - Integer - The number of hidden layer neurons
        oneuron_count  - Integer - The number of output neurons
        eta            - Float   - The Leaning Rate
        max_iterations - Integer - The max number of iterations the MPL will perform while learning
        """

        self.ih_weights = np.random.rand(ineuron_count + 1, hneuron_count) * (2 / np.sqrt(ineuron_count + 1)) - (1 / np.sqrt(ineuron_count) + 1)
        self.ho_weights = np.random.rand(hneuron_count + 1, oneuron_count) * (2 / np.sqrt(hneuron_count + 1)) - (1 / np.sqrt(hneuron_count) + 1)
        self.eta = eta
        self.max_iterations = max_iterations
        self.alpha = momentum

    def activation(self, neuron_values):

        """ Using a sigmoid function determine whether or not a neuron fires """

        activation = lambda x: np.exp

        return np.array([(np.tanh(neuron_values[i]) + 1) / 2 for i in range(neuron_values.size)])

    def eval(self, input):

        """
        Evaluates the Network with the given input

        input - Numpy Array Integers - The input to evaluate
        """

        ## Add bias to the input
        biased_input = np.insert(input, 0, -1)

        ## Evaluate Hidden Layer
        hidden = self.activation(np.matmul(biased_input, self.ih_weights))

        ## Evaluate Ouput Layer
        biased_hidden = np.insert(hidden, 0, -1)
        return self.activation(np.matmul(biased_hidden, self.ho_weights))

    def train(self, inputs, targets):

        """
        Trains the Perceptron on the given inputs with matching targets

        inputs  - Numpy 2D Array Integer - All inputs to train on
        targets - Numpy Array Integer    - Input targets
        """

        ## Initialization
        update_hidden = np.zeros(np.shape(self.ih_weights))
        update_output = np.zeros(np.shape(self.ho_weights))

        ## Ensure Targets is a column vector
        targets = targets[:, np.newaxis]

        for current_iteration in range(self.max_iterations):

            ## Add Bias to inputs
            biased_inputs = np.insert(inputs, 0, -1, axis = 1)

            ## Evaluate Hidden
            hidden = np.matmul(biased_inputs, self.ih_weights)
            hidden = np.array([self.activation(hidden[i,:]) for i in range(hidden.shape[0])])

            ## Evaluate Output
            biased_hidden = np.insert(hidden, 0, -1, axis = 1)
            outputs = np.matmul(biased_hidden, self.ho_weights)
            outputs = np.array([self.activation(outputs[i, :]) for i in range(outputs.shape[0])])

            ## Calculate Changes
            delta_output = (targets - outputs) * outputs * (1.0 - outputs)
            delta_hidden = biased_hidden * (1.0 - biased_hidden) * np.matmul(delta_output, np.transpose(self.ho_weights))

            ## Calculate Updates
            update_hidden = self.eta * np.matmul(np.transpose(biased_inputs), delta_hidden[:,:-1]) + self.alpha * update_hidden
            update_output = self.eta * np.matmul(np.transpose(biased_hidden), delta_output) + self.alpha * update_output

            ## Apply Updates
            self.ih_weights += update_hidden
            self.ho_weights += update_output


    def train_recur(self, inputs, targets, current_iteration):

        """
        Recursive portion of the 'train' method

        inputs            - Numpy 2D Array Integer - All inputs to train on
        targets           - Numpy Array Integer    - Input targets
        current_iteration - Integer                - Current Training Iteration
        """





if __name__ == '__main__':

    ## Data
    and_data = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    xor_data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    ## Perceptron
    perceptron = MPL(2, 3, 1, 0.25, 0.9, 1001)

    #print("Hidden Weights\n", perceptron.ih_weights)
    #print("Output Weights\n", perceptron.ho_weights)

    perceptron.train(and_data[:,:-1], and_data[:,-1])

    print(perceptron.eval([0, 0]))
    print(perceptron.eval([1, 0]))
    print(perceptron.eval([0, 1]))
    print(perceptron.eval([1, 1]))
