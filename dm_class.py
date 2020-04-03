from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import confusion_matrix
from dim_methods import get_varimax_loadings_standard as varimax

class dmMethod:
    def __init__(self, data, n_components = None, standardize = False, time_first = True):
        self.dataset = data
        self.n_components = n_components

        if len(self.dataset) > 3:
            self.dataset = self.dataset.reshape(self.dataset.shape[0], -1)

        if time_first:
            self.dataset = self.dataset.transpose()

        if standardize:
            self.dataset = self.standardize_data(self.dataset)

    def perform_pca(self):
        self.dm = PCA(n_components = self.n_components)
        self.dm.fit(self.dataset)
        self.components = self.dm.components_

    def perfrom_varimax(self):
        self.dm = varimax(self.dataset, max_comps=self.n_components)
        self.components = self.dm["weights"].transpose()

    def perform_fastICA(self):
        self.dm = FastICA(n_components=self.n_components)
        self.components = self.dm.fit_transform(self.dataset.transpose())

    @staticmethod
    def standardize_data(data):
        """
        standardize data
        """
        data -= data.mean(axis=0)
        data /= data.std(axis=0)
        return data
