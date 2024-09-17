#!/usr/bin/env python3
"""Module for Class Neuron"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """A class that defines a single neuron performing binary classification"""
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron's predictions"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """calculates one pass of gradient descent on neuron"""
        m = Y.shape[1]
        dz = A - Y
        dw = 1 / m * np.dot(dz, X.T)
        db = 1 / m * np.sum(dz)
        self.__W -= alpha * dw
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neuron using gradient descent"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")
        
        costs = []
        iteration = []
        for i in range(iterations + 1):
            A = self.forward_prop(X)
            cost = self.cost(Y, A)
            if verbose and ((i % step == 0) or (i == iterations)):
                print(f"Cost after {i} iterations: {cost}")
                costs.append(cost)
                iteration.append(i)
            if i < iterations:
                self.gradient_descent(X, Y, A, alpha=alpha)

        if graph:
            plt.plot(np.arange(0, iterations, step), cost)
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
