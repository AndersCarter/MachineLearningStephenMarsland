import numpy as np
from perceptron import Perceptron

## Execution
if __name__ == "__main__":

    # Logical OR inputs and target value
    inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    targets = np.array([[0], [1], [1], [1]])

    # Perceptron
    p = Perceptron()
    p.train(inputs, targets, 0.25, 6)

    print("Eval")
    print(f"0 0: {p.eval([0, 0])}")
    print(f"0 1: {p.eval([0, 1])}")
    print(f"1 0: {p.eval([1, 0])}")
    print(f"1 1: {p.eval([1, 1])}")

    # Logical XOR inputs and target value
    # The left most number is ignored
    inputs = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    targets = np.array([[0], [1], [1], [0]])

    p = Perceptron()
    p.train(inputs, targets, 0.25, 15)

    print("Eval")
    print(f"0 0 1: {p.eval([0, 0, 1])}")
    print(f"0 1 0: {p.eval([0, 1, 0])}")
    print(f"1 0 0: {p.eval([1, 0, 0])}")
    print(f"1 1 0: {p.eval([1, 1, 0])}")
