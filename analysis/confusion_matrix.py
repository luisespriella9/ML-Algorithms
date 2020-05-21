import numpy as np
import matplotlib.pyplot as plt
import math

class ConfusionMatrix:
    def fit(self, prediction_results, actual_results):
        self.prediction_results = prediction_results
        self.actual_results = actual_results
        if (len(self.prediction_results) != len(self.actual_results)):
            print("must be the same length")
            return
        self.labels = np.unique(self.actual_results)
        self.label_count = len(self.labels)
        dict_id = 0
        self.labels_dict = {}
        for label in self.labels:
            self.labels_dict[label] = dict_id
            dict_id+=1
        self.confusion_matrix = [[0 for i in range(self.label_count)] for i in range(self.label_count)]
        
    def measure(self, label):
        '''
        True Positive (TP) : Observation is positive, and is predicted to be positive.
        False Negative (FN) : Observation is positive, but is predicted negative.
        True Negative (TN) : Observation is negative, and is predicted to be negative.
        False Positive (FP) : Observation is negative, but is predicted positive.
        taken from https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
        '''
        measurements_dict = {}
        self.true_positives = np.sum([True if (self.prediction_results[i] == label) and (self.actual_results[i] == label) else False for i in range(len(self.prediction_results))])
        self.true_negatives = np.sum([True if (self.prediction_results[i] != label) and (self.actual_results[i] != label) else False for i in range(len(self.prediction_results))])
        self.false_positives = np.sum([True if (self.prediction_results[i] == label) and (self.actual_results[i] != label) else False for i in range(len(self.prediction_results))])
        self.false_negatives = np.sum([True if (self.prediction_results[i] != label) and (self.actual_results[i] == label) else False for i in range(len(self.prediction_results))])
        measurements_dict["True Positives"] = self.true_positives
        measurements_dict["False Negatives"] = self.false_negatives
        measurements_dict["True Negatives"] = self.true_negatives
        measurements_dict["False Positives"] = self.false_positives
        return measurements_dict
        
    def accuracy_rate(self):
        '''
        Fraction of times the classifier is correct 
        '''
        return (self.true_positives+self.true_negatives)/(self.true_positives+self.true_negatives+self.false_positives+self.false_negatives)
        
    def error_rate(self):
        return 1-self.accuracy_rate()
    
    def recall(self):
        '''
        True Positive Rate
        '''
        return self.true_positives/(self.true_positives+self.false_negatives)
    
    def false_positive_rate(self):
        return self.false_positives/(self.true_negatives+self.false_positives)
    
    def true_negative_rate(self):
        '''Specificity
        '''
        return 1-self.recall()
    
    def precision(self):
        '''
        How often is it correct
        '''
        return self.true_positives/(self.true_positives+self.false_positives)
    
    def prevalence(self):
        '''
        What is the fraction of apples
        '''
        return (self.true_positives+self.false_negatives)/(self.true_positives+self.true_negatives+self.false_positives+self.false_negatives)
        
    def matthews_correlation_coefficient(self):
        '''
        Correlation that measures how good the classifier is between plus and minus one.
        Plus one means perfect prediction
        Zero means no better than random classifier
        '''
        return ((self.true_positives*self.true_negatives)-(self.false_positives*self.false_negatives))/math.sqrt((self.true_positives+self.false_positives)*(self.true_positives+self.false_negatives)*(self.true_negatives+self.false_positives)*(self.true_negatives+self.false_negatives))
    
    def measurements(self, result, label = None):
        if (label == None):
            label = str(result)
        self.measure(result)
        print("Confusion Matrix label measurements for ", label)
        print("Accuracy rate", self.accuracy_rate())
        print("Error rate", self.error_rate())
        print("Recall", self.recall())
        print("False Positive Rate", self.false_positive_rate())
        print("True Negative Rate", self.true_negative_rate())
        print("Precision", self.precision())
        print("Prevalence", self.prevalence())
        print("Matthews Correlation Coefficient", self.matthews_correlation_coefficient())

    def plot(self):
        for i in range(len(self.prediction_results)):
            if self.prediction_results[i] == self.actual_results[i]:
                key_id = self.labels_dict[self.prediction_results[i]]
                self.confusion_matrix[key_id][key_id] += 1
            else:
                true_key_id = self.labels_dict[self.actual_results[i]]
                predicted_key_id = self.labels_dict[self.prediction_results[i]]
                self.confusion_matrix[true_key_id][predicted_key_id] += 1
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        im = ax.imshow(self.confusion_matrix, cmap="PuBuGn")
        for (j,i),label in np.ndenumerate(self.confusion_matrix):
            ax.text(i,j,self.confusion_matrix[j][i],ha='center',va='center', c="black")
        ax.set_xticks(np.arange(len(self.labels)))
        ax.set_xticklabels(self.labels)
        ax.set_yticks(np.arange(len(self.labels)))
        ax.set_yticklabels(self.labels)
        ax.set_xlabel("Predicted Label", color="g", fontsize=16, fontweight='bold')
        ax.set_ylabel("True Label", color="g", fontsize=16, fontweight='bold')
        fig.colorbar(im)
        plt.show()