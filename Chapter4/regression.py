import os
import numpy as np
from MLP import MLP
from bokeh.plotting import figure, output_file, save

if __name__ == "__main__":

    ## Data a sine wave with guassian noise
    x = np.linspace(0, 1, 40).reshape((1, 40))
    t = np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x) + np.random.randn(40) * 0.2
    x = x.T
    t = t.T


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

    ## Bokeh plots
    plot = figure(title = "MLP Random Sin Wave with Guassian Noise Added")

    ## Plot data
    x_dat = np.ndarray.flatten(x.T)
    t_dat = np.ndarray.flatten(t.T)
    test_dat = np.ndarray.flatten(test.T)
    actual_dat = np.ndarray.flatten(testTarget.T)
    eval_dat = np.ndarray.flatten(np.array([p.eval(val) for val in test]))

    ## Actual Data
    plot.line(x_dat, t_dat, legend = "Full Input Sine Wave")
    plot.circle(x_dat, t_dat, legend = "Full Input Sine Wave")
    plot.line(test_dat, actual_dat, legend = "Test Set With Target Values", color = "red")
    plot.circle(test_dat, actual_dat, legend = "Test Set With Target Values",color = "red")
    plot.line(test_dat, eval_dat, legend = "Test Set with MLP Output", color = "green")
    plot.circle(test_dat, eval_dat, legend = "Test Set with MLP Output", color = "green")

    ## Legend
    plot.legend.click_policy = "hide"

    ## Axis Labels
    plot.xaxis.axis_label = "Input"
    plot.yaxis.axis_label = "Target"

    ## Output
    directory = os.path.dirname(os.path.abspath(__file__))
    output_file(os.path.join(directory, "regression.html"), mode = "inline")
    save(plot)
