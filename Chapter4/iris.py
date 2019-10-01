import os
import pandas as pd
import numpy as np
from MLP import MLP

if __name__ == "__main__":

    ## Create Iris.data filepath
    directory = os.path.dirname(os.path.abspath(__file__))
    iris_fp = os.path.join(directory, "iris.data")
    if not os.path.isfile(iris_fp): raise Exeception("This Script requires the download of the Iris Dataset from the UCI Machine Learning repository found here 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/'")

    ## Create Dataframe
    headers = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    iris_df = pd.read_csv(iris_fp, names = headers)

    ## Preprocess
    ## Replace Text with Integer Representations
    iris_df = iris_df.replace("Iris-setosa", 0)
    iris_df = iris_df.replace("Iris-versicolor", 1)
    iris_df = iris_df.replace("Iris-virginica", 2)
    iris = iris_df.values

    ## Normalize all non class columns
    iris[:,:4] = iris[:,:4] - iris[:,:4].mean(axis = 0)
    imax = np.concatenate((iris.max(axis = 0) * np.ones((1, 5)), np.abs(iris.min(axis = 0) * np.ones((1, 5)))), axis = 0).max(axis = 0)
    iris[:,:4] = iris[:,:4] / imax[:4]

    ## Create Target Set
    targets = np.zeros((iris.shape[0], 3))
    targets[np.where(iris[:,4] == 0), 0] = 1
    targets[np.where(iris[:,4] == 1), 1] = 1
    targets[np.where(iris[:,4] == 2), 2] = 1

    ## Randomize targets
    order = list(range(iris.shape[0]))
    np.random.shuffle(order)
    iris = iris[order,:]
    targets = targets[order,:]

    ## Create Training Sets
    train = iris[::2,0:4]
    train_target = targets[::2]
    valid = iris[1::4,0:4]
    valid_target = targets[1::4]
    test = iris[3::4,0:4]
    test_target = targets[3::4]

    ## MLP
    p = MLP(4, 5, 3, activation_type = "softmax")
    p.train(train, train_target, 5001, valid_set = valid, valid_targets = valid_target)
    p.confusion_matrix(test, test_target)
