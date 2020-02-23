import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.colors as colors
colors_list = list(colors._colors_full_map.values())

class KMC:
    def __init__(self, k, random_state = 0, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.labels = None
        self.random_state = random_state
        
    def initialize_centroids(self, train_data):
        if (self.k > len(train_data)):
            print("k cannot be greater than size of data")
            return
        centroids = train_data.sample(self.k, random_state=self.random_state).to_numpy()
        return centroids
        
    def calculate_centroids(self):
        self.train_data['cluster'] = self.cluster_belongings
        new_centroids = []
        for i in range(self.k):
            cluster_k_data = self.train_data[self.train_data['cluster']==i]
            cluster_k_data = cluster_k_data.drop(['cluster'], axis=1)
            new_centroid = []
            for column in cluster_k_data.columns:
                new_centroid.append(np.mean(cluster_k_data[column]))
            new_centroids.append(new_centroid)
        return new_centroids
    
    def assign_clusters(self):
        old_centroids = self.centroids
        total_error = 0
        for i in range(len(self.train_data)):
            sample = self.train_data.iloc[i]
            #calculate closest cluster and its distance
            cluster, distance = self.closest_cluster(old_centroids, sample)
            total_error += distance
            self.cluster_belongings[i] = cluster
        return self.cluster_belongings, total_error
    
    def euclidean_distance(self, point1, point2):
        distance = 0
        for i in range(len(point1)):
            distance += ((point2[i] - point1[i])**2)
        return math.sqrt(distance)
    
    def closest_cluster(self, centroids, sample):
        closest_c = 0
        shortest_distance = self.euclidean_distance(centroids[0], sample)
        for i in range(1, self.k):
            dist = self.euclidean_distance(centroids[i], sample)
            if (dist < shortest_distance):
                closest_c = i
                shortest_distance = dist
        return closest_c, shortest_distance
    
    def fit(self, train_data):
        if ('cluster' in train_data.columns):
            #drop cluster when rerunning
            train_data = train_data.drop(['cluster'], axis=1) 
        self.train_data = train_data
        self.n_features = len(self.train_data.columns)
        #initialize centroids
        self.centroids = self.initialize_centroids(train_data)
        #set all cluster belongings to cluster 0
        self.cluster_belongings = [0 for i in range(len(train_data))]
        for iteration in range(self.max_iter):
            old_centroids = self.centroids
            #assign clusters
            self.cluster_belongings, total_error = self.assign_clusters()
            #recalculate centroids
            self.centroids = self.calculate_centroids()
            if np.all(self.centroids == old_centroids):
                self.assign_clusters()
                return self.cluster_belongings, self.centroids, total_error
        return "error, reached max iterations"
    
    def set_cluster_labels(self, labels):
        if (len(labels) != len(self.centroids)):
            return "Must be the same size!"
        self.labels = labels
            
    def predict(self, test_data):
        self.test_data = test_data
        if (type(test_data) == pd.core.frame.DataFrame):
            test_samples = test_data.values
        else:
            test_samples = test_data
        self.data_predictions = []
        for sample in  test_samples:
            cluster, distance = self.closest_cluster(self.centroids, sample)
            self.data_predictions.append(cluster)
        return self.data_predictions, self.centroids
    
    def plot_fit(self, roationAngle = None):
        fig=plt.figure()
        if (self.n_features > 3):
            return "cannot plot more than 3 dimensions"
        if (self.n_features == 3):
            ax = Axes3D(fig)
            ax.set_xlabel(self.train_data.columns[0])
            ax.set_ylabel(self.train_data.columns[1])
            ax.set_zlabel(self.train_data.columns[2])
            if (roationAngle != None):
                ax.view_init(azim=roationAngle)
        elif (self.n_features == 2):
            ax=fig.add_axes([0,0,1,1])
            ax.set_xlabel(self.train_data.columns[0])
            ax.set_ylabel(self.train_data.columns[1])
            
        self.train_data['cluster'] = self.cluster_belongings

        for i in range(len(self.centroids)):
            cluster_k_data = self.train_data[self.train_data['cluster']==i]
            ax.scatter(cluster_k_data[cluster_k_data.columns[0]], cluster_k_data[cluster_k_data.columns[1]], cluster_k_data[cluster_k_data.columns[2]], c=colors_list[i])

        ax.set_title("Train Results for Iris data")
        if (self.labels != None):
            plt.legend(handles=[mpatches.Patch(color=colors_list[i], label=self.labels[i]) for i in range(len(self.labels))])
        plt.show()
        
    def plot_predictions(self, roationAngle = None):
        fig=plt.figure()
        if (self.n_features > 3):
            return "cannot plot more than 3 dimensions"
        if (self.n_features == 3):
            ax = Axes3D(fig)
            ax.set_xlabel(self.test_data.columns[0])
            ax.set_ylabel(self.test_data.columns[1])
            ax.set_zlabel(self.test_data.columns[2])
            if (roationAngle != None):
                ax.view_init(azim=roationAngle)
        elif (self.n_features == 2):
            ax=fig.add_axes([0,0,1,1])
            ax.set_xlabel(self.test_data.columns[0])
            ax.set_ylabel(self.test_data.columns[1])

        self.test_data['cluster'] = self.data_predictions
        for i in range(len(self.centroids)):
            cluster_k_data = self.test_data[self.test_data['cluster']==i]
            ax.scatter(cluster_k_data[cluster_k_data.columns[0]], cluster_k_data[cluster_k_data.columns[1]], cluster_k_data[cluster_k_data.columns[2]], c=colors_list[i])

        ax.set_title("Test Results for Iris data")
        if (self.labels != None):
            plt.legend(handles=[mpatches.Patch(color=colors_list[i], label=self.labels[i]) for i in range(len(self.labels))])
        plt.show()

    def screePlot(self, train_data, max_clusters= 10):
        errors_per_number_of_cluster = []
        for i in range(1, max_clusters):
            kmc_i = KMC(i)
            centroid_prediction, centroids, total_error = kmc_i.fit(train_data)
            errors_per_number_of_cluster.append(total_error)
        plt.plot([i for i in range(1, max_clusters)], errors_per_number_of_cluster)
        plt.show()
        
    def get_test_prediction_labels(self):
        if (self.labels == None):
            return "no labels to assign"
        test_result_labels = []
        for cluster in self.data_predictions:
            test_result_labels.append(self.labels[cluster])
        return test_result_labels
        
    def set_k(self, k):
        self.k = k