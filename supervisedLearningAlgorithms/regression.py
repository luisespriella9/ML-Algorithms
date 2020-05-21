import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random

class LinearRegression:
    def fit(self, x_train, y_train, alpha = 0.01, iterations = 1000):
        '''
        using gradient descent for faster calculations
        '''
        self.x_train = x_train
        self.y_train = y_train
        self.n = len(x_train) 
        self.alpha = alpha
        self.iterations = iterations
        x_matrix_df = pd.DataFrame([1 for i in range(self.n)], columns=['b_0 index'])
        
        for col in x_train.columns:
            x_matrix_df[col] = x_train[col]
        x_matrix = x_matrix_df.to_numpy()
        
        #Actual expensive function without gradient descent
        #b_matrix = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_matrix), x_matrix)), np.transpose(x_matrix)), y_train)
        self.theta = self.gradientDescent(x_matrix, y_train, [0]*(len(x_train.columns)+1))
        return self.theta
        
    def plot_regression_line(self, y_label = 'Prediction'):
        for i in range(len(self.x_train.columns)):
            plt.scatter(self.x_train[self.x_train.columns[i]], self.y_train, color = "m", 
               marker = "o") 
            y_pred = self.theta[0] + (self.theta[i+1]*self.x_train[self.x_train.columns[i]])
            plt.plot(self.x_train[self.x_train.columns[i]], y_pred, color = "g")
            plt.xlabel(self.x_train.columns[i]) 
            plt.ylabel(y_label) 
            plt.show()
            
    def predict(self, x_test):
        test_size = len(x_test)
        x_test_matrix_df = pd.DataFrame([1 for i in range(test_size)], columns=['b_0 index'])
        for col in x_test.columns:
            x_test_matrix_df[col] = x_test[col]
        x_test_matrix = x_test_matrix_df.to_numpy()
        return np.dot(x_test_matrix, self.theta)
    
    def gradientDescent(self, X, y, theta):
        cost_history = [0] * self.iterations
        self.cost_history = []
        XTrans = np.transpose(X)
        for i in range(self.iterations):
            y_pred = np.dot(X, theta)
            loss = y_pred - y
            gradient = np.dot(XTrans, loss) 
            theta -= ((self.alpha/ self.n) * gradient)
            if (np.isnan(theta).any() or np.isinf(theta).any()):
                break
            cost = self.costFunction(loss, self.n)
            self.cost_history.append((cost, theta))
        return min(self.cost_history, key = lambda t: t[0])[1]

    def plotCostPerIteration(self):
        costs = [c[0] for c in self.cost_history]
        plt.plot(np.arange(len(self.cost_history)), costs)
        plt.xlabel("iterations") 
        plt.ylabel("cost") 
        plt.show()
        
    def costFunction(self, loss, m):
        J = np.sum((loss) ** 2)/(2 * m)
        return J

class LogisticRegression:
    def fit(self, x_train, y_train, alpha = 0.000001, iterations = 1000):
        '''
        using gradient descent for faster calculations
        '''
        self.x_train = x_train
        self.y_train = y_train
        self.alpha = alpha
        self.iterations = iterations
        self.m = len(x_train) 
        self.x_matrix = x_train.to_numpy()
        self.XTrans = np.transpose(self.x_matrix)
        theta_default = [0]*(len(self.x_train.columns))
        self.theta = self.gradientDescent(self.x_matrix, self.y_train, theta_default)
        return self.theta
    
    def plot(self, y_label = 'Prediction'):
        for i in range(len(self.x_train.columns)):
            plt.scatter(self.x_train[self.x_train.columns[i]], self.y_train, color = "m", 
               marker = "o") 
            y_pred = self.computePrediction(self.x_train, self.theta)
            plt.scatter(self.x_train[self.x_train.columns[i]], y_pred, color = "g")
            plt.xlabel(self.x_train.columns[i]) 
            plt.ylabel(y_label) 
            plt.show()
    
    def computePrediction(self, x, theta):
        scores = np.dot(x, theta)
        return self.sigmoid(scores)
    
    def predict(self, x_test):
        return self.computePrediction(x_test, self.theta)
    
    def gradientDescent(self, X, y, theta):
        cost_history = [0] * self.iterations
        self.cost_history = []
        for i in range(self.iterations):
            y_pred = self.computePrediction(X, theta)
            loss = y_pred - y
            gradient = np.dot(self.XTrans, loss) 
            theta -= (self.alpha * gradient)
            if (np.isnan(theta).any() or np.isinf(theta).any()):
                break
            cost = self.costFunction(y_pred, y)
            self.cost_history.append((cost, theta))
        return min(self.cost_history, key = lambda t: t[0])[1]
    
    def sigmoid(self, scores):
        return 1.0 / (1+np.exp(-scores))
        
    def costFunction(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()