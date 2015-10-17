__author__ = 'wangyichao'

from sklearn.neighbors import NearestNeighbors

# samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5], [3, 3, 3]]
# neigh = NearestNeighbors(n_neighbors=2, return_distance=False)
# neigh.fit(samples)
# print(neigh.kneighbors([1., 1., 1.]))


NUM_NEIGHBORS = 5

class KnnManager():

    def __init__(self):
        self.neigh = NearestNeighbors(n_neighbors=NUM_NEIGHBORS, return_distance=False)

    def fit(self, train_list):
        self.neigh.fit(train_list)

    def query(self, query_list):
        return self.neigh.kneighbors(query_list)

