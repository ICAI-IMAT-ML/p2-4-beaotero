import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            return self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session

        # Store the intercept and the coefficients of the model
        Xtras = np.transpose(X)
        term1 = np.linalg.inv(np.dot(Xtras, X))
        w = np.dot(np.dot(term1, Xtras), y)

        self.intercept = w[0]
        self.coefficients = w[1:]

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01
        ephocs = []
        mses = []
        w_history = []
        b_history = []
        # Implement gradient descent (TODO)
        for epoch in range(iterations):
            predictions = self.predict(X[:, 1:])
            error = predictions - y

            # TODO: Write the gradient values and the updates for the paramenters
            gradient = 1/m * np.dot(error, X)

            self.intercept -= learning_rate*gradient[0]
            self.coefficients -= learning_rate*gradient[1:]
            w_history.append(self.coefficients[0])  # Suponiendo 1 sola feature
            b_history.append(self.intercept)
        
            # TODO: Calculate and print the loss every 10 epochs
            if epoch % 1000 == 0:
                mse = np.sum(error ** 2)/m
                print(f"Epoch {epoch}: MSE = {mse}")
                ephocs.append(epoch)
                mses.append(mse)
        return ephocs, mses, w_history, b_history
        

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")
        if np.ndim(X) == 1:
            predictions = X * self.coefficients + self.intercept
        else:
            predictions = X.dot(self.coefficients) + self.intercept
            
        return np.array(predictions)
        


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """
    n = len(y_true)
    
    # R^2 Score
    n = len(y_true)
    suma1 =np.sum((y_true-y_pred)**2)
    suma2 =np.sum((y_true-np.mean(y_true))**2)
    r_squared = 1-(suma1/suma2)

    # Root Mean Squared Error
   
    suma =np.sum(np.power(y_pred-y_true,2))
    rmse = np.sqrt(1/n *suma)

    # Mean Absolute Error
    
    suma = np.sum(abs(y_pred-y_true))
    mae = 1/n * suma

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    columns_added = 0
    for index in sorted(categorical_indices, reverse=True):
        # TODO: Extract the categorical column
        categorical_column = X[:, index]

        unique_values = set(categorical_column)
        
        one_hot = np.array([[ 1 if y == x else 0 for y in unique_values ] for x in categorical_column])

        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]

        index += columns_added
        X_transformed = np.delete(X_transformed,index,1)
        X_transformed = np.concatenate((one_hot,X_transformed),1)
        columns_added += one_hot.shape[1] 

    return X_transformed
