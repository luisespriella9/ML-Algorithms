import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def fit(self, x_train, y_train, alpha = 0.01, regularization = 0.1, epochs=1000):
        '''
        using gradient descent for faster calculations
        '''
        # reshape x
        m_samples, n_features = x_train.shape
        x_train = x_train.T # X should have a sample per column X shape: (n features, m samples)
        x_train = np.insert(x_train, 0, np.ones((1, m_samples)), 0) 
        n_features = n_features + 1
        # randomize theta
        
        theta = np.random.rand(n_features, 1)
        self.cost_history = []
        # fit data
        for epoch in range(epochs):
            predictions = []
            for i in range(m_samples):
                gradient, prediction = self.gradient(x_train[:, i], y_train[i], theta, regularization)
                predictions.append(prediction)
                theta = theta - (alpha*gradient)
            cost = self.cost(predictions, y_train, theta, regularization)
            self.cost_history.append(cost)
                
        # save to model
        self.theta = theta
        self.x_train = x_train
        return self.theta
        
            
    def predict(self, x_test):
        m_samples, _ = x_test.shape
        x_test = x_test.T # X should have a sample per column X shape: (n features, m samples)
        x_test = np.insert(x_test, 0, np.ones((1, m_samples)), 0) 
        prediction = np.dot(self.theta.T, x_test)
        return prediction[0]
    
    def gradient(self, x, y, theta, regularization):
        prediction = np.dot(theta.T, x)[0]
        error = prediction - y
        x = x.reshape(-1, 1)
        theta_grad = np.insert(theta[1:], 0, 0, 0)*regularization
        gradient = np.dot(x, error)
        return gradient, prediction
        
    def cost(self, predictions, y, theta, regularization):
        return np.sum(np.square(predictions-y))/(2*len(y)) + (np.sum(np.square(theta[1:]))*regularization)/(2*len(y))
    
    def plot_cost_per_epoch(self):
        plt.plot(np.arange(0, len(self.cost_history), 1), self.cost_history)

class LogisticRegression:
    def fit(self, x_train, y_train, alpha = 0.01, regularization = 0.1, epochs=100):
        '''
        using gradient descent for faster calculations
        '''
        # reshape x
        m_samples, n_features = x_train.shape
        x_train = x_train.T # X should have a sample per column X shape: (n features, m samples)
        x_train = np.insert(x_train, 0, np.ones((1, m_samples)), 0) 
        n_features = n_features + 1
        # randomize theta
        
        theta = np.random.rand(n_features, 1)
        self.cost_history = []
        # fit data
        for epoch in range(epochs):
            predictions = []
            for i in range(m_samples):
                gradient, prediction = self.gradient(x_train[:, i], y_train[i], theta, regularization)
                predictions.append(prediction)
                theta = theta - (alpha*gradient)
            cost = self.cost(predictions, y_train, theta, regularization)
            self.cost_history.append(cost)
                
        # save to model
        self.theta = theta
        self.x_train = x_train
        return self.theta
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
        
            
    def predict(self, x_test):
        m_samples, _ = x_test.shape
        x_test = x_test.T # X should have a sample per column X shape: (n features, m samples)
        x_test = np.insert(x_test, 0, np.ones((1, m_samples)), 0) 
        prediction = self.sigmoid(np.dot(self.theta.T, x_test))
        return prediction[0]
    
    def gradient(self, x, y, theta, regularization):
        prediction = self.sigmoid(np.dot(theta.T, x))[0]
        error = prediction - y
        x = x.reshape(-1, 1)
        theta_grad = np.insert(theta[1:], 0, 0, 0)*regularization
        gradient = np.dot(x, error)
        return gradient, prediction
        
    def cost(self, predictions, y, theta, regularization):
        return -np.sum(y*np.log(predictions)+((1-y)*np.log([1-p for p in predictions])))/len(y)
    
    def plot_cost_per_epoch(self):
        plt.plot(np.arange(0, len(self.cost_history), 1), self.cost_history)