import numpy as np

class MinMaxScaler:
    def __init__(self, feature_range = (0, 1)):
        self.feature_range = feature_range
    '''
    X_new = Xi-min(X)/(max(X)-min(X)) 
    '''
    def fit_transform(self, df):
        self.mins = {}
        self.max = {}
        self.columns = df.columns
        for column in self.columns:
            feature_min = np.min(df[column].values)
            self.mins[column] = feature_min
            feature_max = np.max(df[column].values)
            self.max[column] = feature_max
        return self.transform(df)

    def transform(self, test_data):
        standarized_df = test_data.copy() 
        for column in self.columns:
            standarized_df[column] = test_data[column].apply(self.scaleFeature, feature_min = self.mins[column], feature_max = self.max[column])
        return standarized_df

        
    def scaleFeature(self, feature_val, feature_min, feature_max):
        return ((feature_val-feature_min)*(self.feature_range[1]-self.feature_range[0])/(feature_max-feature_min))
    

class StandardScaler():
    '''
    X_new = Xi-X_mean/standard_deviation
    '''
    def fit_transform(self, df):
        self.means = {}
        self.deviations = {}
        self.columns = df.columns
        for column in df.columns:
            feature_mean = np.mean(df[column].values)
            self.means[column] = feature_mean
            feature_standard_deviation = np.std(df[column].values)
            self.deviations[column] = feature_standard_deviation
        return self.transform(df)

    def transform(self, test_data):
        standarized_df = test_data.copy() 
        for column in self.columns:
            standarized_df[column] = test_data[column].apply(self.scaleFeature, feature_mean = self.means[column], feature_standard_deviation = self.deviations[column])
        return standarized_df
    
    def scaleFeature(self, feature_val, feature_mean, feature_standard_deviation):
        return (feature_val-feature_mean)/feature_standard_deviation