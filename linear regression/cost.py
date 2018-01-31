import numpy as np
from util import *

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))

    J = -1.0 * (1.0 / m) * (np.log(h + eps).T.dot(y) + np.log(1 - h + eps).T.dot(1 - y))


    if np.isnan(J[0]):
        return (np.inf)
    return J[0]

def costFunctionReg(theta, reg, XX, y, *args):
    m = y.size
    h = sigmoid(XX.dot(theta))
    
    J = -1.0*(1.0/m)*(np.log(h + eps).T.dot(y)+np.log(1-h + eps).T.dot(1-y)) + (reg/(2.0*m))*np.sum(np.square(theta[1:]))
    
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])

def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))

    grad = (1.0 / m) * X.T.dot(h - y)

    return (grad.flatten())

def gradientReg(theta, reg, XX, y, *args):
    m = y.size
    h = sigmoid(XX.dot(theta.reshape(-1,1)))
      
    grad = (1.0/m)*XX.T.dot(h-y) + (reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
        
    return(grad.flatten())

def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return (p.astype('int'))
