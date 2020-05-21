import pandas as pd
import numpy as np

class NaiveBayes():
        
    def fit(self, x_train, results):
        self.results = results
        self.results_unique = np.unique(results)
        self.results_probs = ["P(" + prob + ")" for prob in self.results_unique]
        self.feature_tables = {}
        for column in x_train.columns:
            all_values = []
            for val in x_train[column]:
                if (val not in all_values):
                    all_values.append(val)
            feature_df = pd.DataFrame(0, columns = self.results_unique, index = all_values)
            self.feature_tables[column] = feature_df
        for i in range(len(x_train)):
                row = x_train.iloc[i, :]
                classification = results[i]
                for feature in row.keys():
                    self.feature_tables[feature].ix[row[feature]][results[i]]+=1
                
        for k, df in self.feature_tables.items():
            df['Total'] = 0
            for row_index in df.index:
                df.loc[row_index, 'Total'] = np.sum(df.loc[row_index].values)
                
        for k, df in self.feature_tables.items():
            for column in df.columns:
                prob_col_title = "P(" + column + ")"
                df[prob_col_title] = df[column]/np.sum(df[column].values)
            
        for k, df in self.feature_tables.items():
            df.ix['Total'] = 0
            for column in df.columns:
                df.loc['Total', column] = np.sum(df[column].values)
                
    def predict(self, x_test):
        predictions = []
        for i in range(len(x_test)):
            sample = x_test.iloc[i, :]
            prediction, probabilities = self.predict_sample(sample)
            predictions.append(prediction)
        return predictions
    
    def predict_probability(self, x_test, label):
        probability_scores = []
        for i in range(len(x_test)):
            sample = x_test.iloc[i, :]
            prediction, probabilities = self.predict_sample(sample)
            probability_scores.append(probabilities[label])
        return probability_scores
            
    def predict_sample(self, sample):
        probabilities = {}
        for column in self.results_probs:
            probs_given_result = 1
            feature_prob = 1
            for feature in sample.index:
                if (sample[feature] not in self.feature_tables[feature].index):
                    continue
                probs_given_result *= self.feature_tables[feature].ix[sample[feature], column]
                feature_prob *= self.feature_tables[feature].ix[sample[feature], 'P(Total)']
            stripped_column = column[2:-1]
            class_prob = np.count_nonzero(self.results == stripped_column)/len(self.results)
            probabilities[stripped_column] = (probs_given_result*class_prob)/feature_prob
        return max(probabilities, key=probabilities.get), probabilities