import numpy as np
import matplotlib.pyplot as plt

class ROC:
    def plot(self, label, prediction_probabilities, actual_results, threshold = [0, .2, .4, .6, .8, 1], color="blue"):
        fpr, tpr = self.measure(label, prediction_probabilities, actual_results, threshold)
        plt.figure()
        plt.plot(fpr, tpr, marker='.', label=label, color=color)
        plt.xlabel("False Positive Rate", color="black", fontsize=12)
        plt.ylabel("True Positive Rate", color="black", fontsize=12)
        plt.legend()
        
    def measure(self, label, prediction_probabilities, actual_results, threshold):
        '''
        True Positive (TP) : Observation is positive, and is predicted to be positive.
        False Negative (FN) : Observation is positive, but is predicted negative.
        True Negative (TN) : Observation is negative, and is predicted to be negative.
        False Positive (FP) : Observation is negative, but is predicted positive.
        taken from https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
        '''
        actual_results = [1 if (actual_result==label) else 0 for actual_result in actual_results]
        tpr_list = []
        fpr_list = []
        for t in threshold:
            t_probs = [1 if (prediction_probabilities[i] >= t) else 0 for i in range(len( prediction_probabilities))]
            true_positives = np.sum([1 if (t_probs[i] == 1) and (actual_results[i] == 1) else False for i in range(len(actual_results))])
            true_negatives = np.sum([1 if (t_probs[i] != 1) and (actual_results[i] != 1) else False for i in range(len(actual_results))])
            false_positives = np.sum([1 if (t_probs[i] == 1) and (actual_results[i] != 1) else False for i in range(len(actual_results))])
            false_negatives = np.sum([1 if (t_probs[i] != 1) and (actual_results[i] == 1) else False for i in range(len(actual_results))])
            tpr = self.recall(true_positives, false_negatives)
            fpr = self.false_positive_rate(false_positives, true_negatives)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        tpr_list.append(0)
        fpr_list.append(0)
        return fpr_list, tpr_list
    def recall(self, true_positives, false_negatives):
        '''
        True Positive Rate
        '''
        recall = true_positives/(true_positives+false_negatives)
        return recall
    
    def false_positive_rate(self, false_positives, true_negatives):
        fpr = false_positives/(true_negatives+false_positives)
        return fpr