import numpy as np

class MinMaxScaler:
    def __init__(self, feature_range = (0, 1)):
        self.feature_range = feature_range
    '''
    X_new = Xi-min(X)/(max(X)-min(X)) 
    '''
    def fit_transform(self, df):
        standarized_df = df.copy() 
        for column in df.columns:
            feature_min = np.min(df[column].values)
            feature_max = np.max(df[column].values)
            standarized_df[column] = df[column].apply(self.scaleFeature, feature_min = feature_min, feature_max = feature_max)
        return standarized_df
        
    def scaleFeature(self, feature_val, feature_min, feature_max):
        return ((feature_val-feature_min)*(self.feature_range[1]-self.feature_range[0])/(feature_max-feature_min))
    

class StandardScaler():
    '''
    X_new = Xi-X_mean/standard_deviation
    '''
    def fit_transform(self, df):
        standarized_df = df.copy() 
        for column in df.columns:
            feature_mean = np.mean(df[column].values)
            feature_standard_deviation = np.std(df[column].values)
            standarized_df[column] = df[column].apply(self.scaleFeature, feature_mean = feature_mean, feature_standard_deviation = feature_standard_deviation)
        return standarized_df
    
    def scaleFeature(self, feature_val, feature_mean, feature_standard_deviation):
        return (feature_val-feature_mean)/feature_standard_deviation