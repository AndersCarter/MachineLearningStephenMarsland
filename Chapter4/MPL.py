import numpy as np

class MPL:

    """ Represents a Multi-layered Perceptron with a single hidden layers"""

    def __init__(self, i_count, h_count, o_count, **kwargs):

        """
        Initializes a new MPL Object

        Arguments
        i_count | Integer : The number of neurons in the input layer
        h_count | Integer : The number of neurons in the hidden layer
        o_count | Integer : The number of neurons in the output layer

        Kwargs
        beta            | Float : Beta value to use in sigmoid activation functions
        momentum        | Float : How much momentum to apply to the weight changes as they are updated
        activation_type | String : The output layer activation function type
        """

        self.beta = kwargs["beta"] if "beta" in kwargs else 1.0
        self.momentum = kwargs["momentum"] if "momentum" in kwargs else 0.9
        self.activation_type = kwargs["activation_type"] if "activation_type" in kwargs else "logistic"

        ## Intitalize Weights
        self.input_weights = (np.random.rand(i_count + 1, h_count) - 0.5) * 2 / np.sqrt(i_count)
        self.hidden_weights = (np.random.rand(h_count + 1, o_count) - 0.5) * 2 / np.sqrt(h_count)

if __name__ == "__main__":

    p = MPL(2, 3, 1)
    print(p.input_weights)
    print(p.hidden_weights)
