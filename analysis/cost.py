import numpy as np

def cost_function(predictions, results):
    '''
    OLS Default cost function
    '''
    residual_error = results-predictions
    return sum([e**2 for e in residual_error])/(2*(len(predictions)))

def sigmoidal_cost_function(y_pred, y):
    return (-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)).mean()