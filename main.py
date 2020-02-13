# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:59:51 2020
Linear Regression with one variable

@author: Ferdi
"""
import numpy as np
from matplotlib import pyplot as plt

file = open("ex1data1.txt", "r")

X = np.array([])
y = np.array([])

for line in file:
    a, b = line.split(',')
    a = float(a)
    b = float(b)
    X = np.append(X, a)
    y = np.append(y, b)

file.close()

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)  
plt.plot(X, y, 'x')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

def compute_cost(X, y, theta):
    m = y.shape[0]
    J = np.sum(np.square(np.matmul(X, theta) - y))/(2*m)
    return J

def gradient_descent(X, y, theta, alpha, iterations):
    J_history = np.empty((iterations, 1))
    for i in range(iterations):
        theta = theta - alpha*(1/m)*np.sum(((np.matmul(X, theta) - y)*X).T, 1).reshape(-1, 1)
        J_history[i][0] = compute_cost(X, y, theta) 
        if i%100 == 0:
            plt.plot(X[:, 1], y, 'x')
            plt.xlabel('Population of City in 10,000s')
            plt.ylabel('Profit in $10,000s')
            plt.plot(X[:, 1], np.matmul(X, theta))
            plt.show()
            input("Press Enter to continue...")
    plt.plot(J_history)
    plt.show()
    return theta

#%%

z = np.ones_like(X)
X = np.hstack((z, X))
theta = np.ones((2, 1))
theta[0][0] = -1
theta[1][0] = -1

iterations = 1500
alpha = 0.01

J = compute_cost(X, y, theta)

theta = gradient_descent(X, y, theta, alpha, iterations);

plt.plot(X[:, 1], y, 'x')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:, 1], np.matmul(X, theta))
plt.show()