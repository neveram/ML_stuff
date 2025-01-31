Linear Regression (Forward)
Your task is to implement linear regression, a statistical model that ends up being the foundation of neural networks. You can learn more from the Complete Explanation of Linear Regression or by reading the description below.

Your must implement get_model_prediction() which returns a prediction value for each dataset value, and get_error() which calculates the error for given prediction data.

Inputs - get_model_prediction:

X - the dataset to be used by the model to predict the output. len(X) = n, and len(X[i]) = 3 for 0 <= i < n.
weights - the current w1,w2,w3 weights for the model. len(weights) = 3.

Inputs - get_error:

model_prediction - the model's prediction for each training example. len(model_prediction) = n.
ground_truth - the correct answer for each example. len(ground_truth) = n.

import numpy as np
from numpy.typing import NDArray


# Helpful functions:
# https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
# https://numpy.org/doc/stable/reference/generated/numpy.mean.html
# https://numpy.org/doc/stable/reference/generated/numpy.square.html

class Solution:
    
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is an Nx3 NumPy array
        # weights is a 3x1 NumPy array
        # HINT: np.matmul() will be useful
        # return np.round(your_answer, 5)
        res= np.matmul(X,weights)
        return np.round(res,5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # model_prediction is an Nx1 NumPy array
        # ground_truth is an Nx1 NumPy array
        # HINT: np.mean(), np.square() will be useful
        # return round(your_answer, 5)
        diff= model_prediction-ground_truth
        sq=np.square(diff)
        mse=np.mean(sq)
        return np.round(mse,5)
