from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


class KNN(NearestNeighbors):
    """
    Add a 'predict' method to the NearestNeighbors class so that it can be used in an sklearn Pipeline
    """
    def predict(self, X, k_neighbors=10):
        return self.kneighbors(X, n_neighbors=k_neighbors)

def similar_bonds_pipeline():
    """
    This is just an example pipeline, feel free to add/remove steps to your liking!
    """
    pipeline = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            #('encoder', OneHotEncoder()),
            ('pca', PCA(n_components=3)),
            ('knn', KNN()),
        ]
    )
    return pipeline

