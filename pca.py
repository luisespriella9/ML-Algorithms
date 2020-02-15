from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
colors_list = list(colors._colors_full_map.values())

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.id_dict = {}
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def fit(self, data):
        self.dim_mean_list = self.mean_per_dimension(data)
    
    def transform(self, data):
        self.centered_data = self.center_columns(data, self.dim_mean_list)
        cov_matrix, cov_matrix_df = self.covariance(data, self.dim_mean_list)
        eigenvector_list = self.eigenvector(cov_matrix, data.columns)[0:self.n_components]
        self.eigenvectors = [eigen[2] for eigen in eigenvector_list]
        self.pca_data = pd.DataFrame(np.dot(self.centered_data, np.transpose(self.eigenvectors)), columns=["Principal Component " + str(i) for i in range(1, self.n_components+1)])
        return self.pca_data
    
    def mean_per_dimension(self, data):
        dim_mean_list = {}
        for column_name, column_data in data.iteritems():
            dim_mean_list[column_name] = np.mean(column_data)
        return dim_mean_list
        
    def covariance(self, data, dim_mean_list):
        num_features = data.shape[1]
        n_samples = data.shape[0]
        cov_matrix = [[0 for i in range(num_features)] for j in range(num_features)]
        col_index = 0
        for col_1 in data.columns:
            row_index = 0
            for col_2 in data.columns:
                cov_matrix[row_index][col_index] = 1/(n_samples-1)*np.sum((data[col_1]-dim_mean_list[col_1])*(data[col_2]-dim_mean_list[col_2]))
                row_index+=1
            col_index += 1
        cov_matrix_df =  pd.DataFrame(cov_matrix, columns=data.columns)
        cov_matrix_df.index = data.columns
        return cov_matrix, cov_matrix_df
    
    def eigenvector(self, cov_matrix, columns):
        eigenvalues, eigenvector = np.linalg.eig(cov_matrix)
        eigenvector_t = np.transpose(eigenvector)
        eigen_tuple = [(columns[i], eigenvalues[i], eigenvector_t[i]) for i in range(len(eigenvalues))]
        #sort by eigenvalue
        return sorted(eigen_tuple, key = lambda x: x[1], reverse=True)   
    
    def center_columns(self, data, mean_per_column):
        centered_df = data.copy()
        for column_name in mean_per_column:
            centered_df[column_name] = centered_df[column_name].apply(self.substract_values, value=mean_per_column[column_name])
        return centered_df.values
        
    def substract_values(self, col, value):
        return col-value
    
    def labelsToInt(self, labels):
        colors = ['r']
        n_labels = len(labels)
        id_result = [0 for i in range(n_labels)]
        id_count = 0
        for i in range(n_labels):
            if labels[i] not in self.id_dict:
                self.id_dict[labels[i]] = id_count
                id_count+=1
            id_result[i] = colors_list[self.id_dict[labels[i]]]
        return id_result, [k for k in self.id_dict]
    
    def plot(self, title, results):
        res_color, labels = self.labelsToInt(results)
        fig=plt.figure()
        if (self.n_components == 2):
            ax=fig.add_axes([0,0,1,1])
            ax.scatter(self.pca_data[self.pca_data.columns[0]], self.pca_data[self.pca_data.columns[1]], c=res_color)
            ax.legend()
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
        elif (self.n_components == 3):
            ax = Axes3D(fig)
            ax.scatter(self.pca_data[self.pca_data.columns[0]], self.pca_data[self.pca_data.columns[1]], self.pca_data[self.pca_data.columns[2]],  c=res_color)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_zlabel("Principal Component 3")
        plt.legend(handles=[mpatches.Patch(color=colors_list[i], label=labels[i]) for i in range(len(labels))])
        ax.set_title(title)
        plt.show()