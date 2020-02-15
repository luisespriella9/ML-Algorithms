import math

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, scaled_data, data_labels):
        self.scaled_data = scaled_data
        self.data_labels = data_labels
        
    def predict(self, sample):
        distances = []
        for i in range(len(self.scaled_data)):
            distance = self.euclidean_distance(self.scaled_data.iloc[i], sample)
            distances.append((distance, self.data_labels[i]))
        distances.sort(key = lambda x: x[0])
        return self.getMaxCount([d[1] for d in distances[:self.k]])
    
    def predict_probability(self, sample, label):
        distances = []
        for i in range(len(self.scaled_data)):
            distance = self.euclidean_distance(self.scaled_data.iloc[i], sample)
            distances.append((distance, self.data_labels[i]))
        distances.sort(key = lambda x: x[0])
        label_appearances = 0
        for closest_point in distances[:self.k]:
            if closest_point[1] == label:
                label_appearances+=1
        return label_appearances/self.k
        
    def getMaxCount(self, l):
        count_dict = {}
        for classification in l:
            if (classification not in count_dict.keys()):
                count_dict[classification] = 1
            else:
                count_dict[classification] += 1
        return max(count_dict, key=count_dict.get)
    
    def euclidean_distance(self, point1, point2):
        distance = 0
        for i in range(len(point1)):
            distance += ((point2[i] - point1[i])**2)
        return math.sqrt(distance)