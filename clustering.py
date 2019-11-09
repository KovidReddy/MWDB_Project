import numpy as np
import copy
import math

# Class for the K-Means Algorithm
class KMeans:
    def __init__(self, k=2, max_iter=500):
        self.k = k
        self.max_iter = max_iter

    # Distance using normalization
    def dist(self,x, y):
        return np.linalg.norm(x - y)

    # Euclidean Distance
    def eucdist(self,x, y):
        sum = 0
        z = x - y
        for el in z:
            sum = sum + el ** 2
        res = math.sqrt(sum)
        return res

    def fit(self,data):
        data_temp = list(copy.deepcopy(data))
        centroids = []
        # Randomly assign the centroids from the data
        for _ in range(self.k):
            x = np.random.randint(0,len(data_temp))
            centroids.append(data_temp[x])
            del data_temp[x]
        centroids = np.array(centroids, dtype=np.float_)
        cent_old = np.zeros(centroids.shape, dtype=np.float_)
        clusters = np.zeros(len(data))

        error = self.dist(centroids, cent_old)

        while (error != 0.0) and (self.max_iter != 0):
            # Assign each data point to the closest centroid
            for i in range(len(data)):
                distances = np.array([self.eucdist(data[i],centroid) for centroid in centroids])
                cluster = np.argmin(distances)
                clusters[i] = cluster
            # Take a copy of the centroids
            cent_old = copy.deepcopy(centroids)

            # Update the centroid list
            for i in range(self.k):
                points = [data[j] for j in range(len(data)) if clusters[j] == i]
                if points:
                    centroids[i] = np.mean(points, axis=0)

            # Calculate the max error
            error = self.dist(centroids, cent_old)
            self.max_iter = self.max_iter - 1

        # Return the Centroids and Clusters
        return centroids, clusters