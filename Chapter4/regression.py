import os
import numpy as np
from MLP import MLP
from bokeh.plotting import figure, output_file, save

def plot_function(x, t):

    """ Plot the function the MLP will try to learn """

    ## Plot Figure
    p = figure(title = "MLP DataSet")
    p.circle(x, t)
    p.line(x, t)
    p.xaxis.axis_label = "x"
    p.yaxis.axis_label = "t"

    ## Output Filepath
    directory = os.path.dirname(os.path.abspath(__file__))
    fp = os.path.join(directory, "regression.html")

    output_file(fp, mode = "inline")
    save(p)


if __name__ == "__main__":

    ## Data a sine wave with guassian noise
    x = np.linspace(0, 1, 40).reshape((1, 40))
    t = np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x) + np.random.randn(40) * 0.2
    x = x.T
    t = t.T

    ## PLot Data
    plot_function(np.ndarray.flatten(x.T), np.ndarray.flatten(t.T))

    ## Data Sets
    train = x[0::2,:]
    test = x[1::4,:]
    valid = x[3::4,:]
    trainTarget = t[0::2,:]
    testTarget = t[1::4,:]
    validTarget = t[3::4,:]

    ## MLP
    p = MLP(1, 3, 1, activation_type = "linear")
    p.train(train, trainTarget, 10001, valid_set = valid, valid_targets = validTarget)
