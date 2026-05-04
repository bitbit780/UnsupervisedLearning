import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import FastICA

from DimReducerBase import DimReducerBase

class PCAReducer(DimReducerBase):
    def get_model(self, **params):
        return PCA(n_components=params.get('n_components', 784),
                   whiten=params.get('whiten', False),
                   random_state=params.get('random_state', 2018))
    
class IncrementalPCAReducer(DimReducerBase):
    def get_model(self, **params):
        return IncrementalPCA(n_components=params.get('n_components', 784),
                              batch_size=params.get('batch_size', None))

class SparsePCAReducer(DimReducerBase):
    def get_model(self, **params):
        return SparsePCA(n_components=params.get('n_components', 100),
                         alpha=params.get('alpha', 0.0001),
                         random_state=params.get('random_state', 2018),
                         n_jobs=params.get('n_jobs', -1))

class KernelPCAReducer(DimReducerBase):
    def get_model(self, **params):
        return KernelPCA(n_components=params.get('n_components', 100),
                         kernel=params.get('kernel', 'rbf'),
                         gamma=params.get('gamma', None),
                         n_jobs=params.get('n_jobs', 1),
                         random_state=params.get('random_state', 2018))

class SingularValueComosition(DimReducerBase):
    def get_model(self, **params):
        return TruncatedSVD(n_components=params.get('n_components',200),
                            algorithm=params.get('algorithm', 'randomized'),
                            n_iter=params.get('n_iter', 5),
                            random_state=params.get('random_state', 2018))
    
class GaussianRandomProjectionReducer(DimReducerBase):
    def get_model(self, **params):
        return GaussianRandomProjection(n_components=params.get('n_components', 'auto'),
                                        eps=params.get('eps', 0.5),
                                        random_state=params.get('random_state', 2018))

class SparseRandomProjectionReducer(DimReducerBase):
    def get_model(self, **params):
        return SparseRandomProjection(n_components=params.get('n_components', 'auto'),
                                      density=params.get('density', 'auto'),
                                      eps=params.get('eps', 0.5),
                                      dense_output=params.get('dense_output', False))

class IsomapReducer(DimReducerBase):
    def get_model(self, **params):
        return Isomap(n_neighbors=params.get('n_neighbors', 5),
                      n_components=params.get('n_components', 10),
                      n_jobs=params.get('n_jobs', 4))

class MultidimentionalScalingReducer(DimReducerBase):
    def get_model(self, **params):
        return MDS(n_components=params.get('n_components', 2),
                   n_init=params.get('n_init', 12),
                   max_iter=params.get('max_iter', 1200),
                   metric=params.get('metric', True),
                   n_jobs=params.get('n_jobs', 4),
                   random_state=params.get('random_state', 2018))

class LLE(DimReducerBase):
    def get_model(self, **params):
        return LocallyLinearEmbedding(n_neighbors=params.get('n_neighbors', 10),
                                      n_components=params.get('n_components', 2),
                                      method=params.get('method', 'modified'),
                                      n_jobs=params.get('n_jobs', 4),
                                      random_state=params.get('random_state', 2018))

class TSNEReducer(DimReducerBase):
    def get_model(self, **params):
        return TSNE(n_components=params.get('n_components', 2),
                    learning_rate=params.get('learning_rate', 300),
                    perplexity=params.get('perplexity', 30),
                    early_exaggeration=params.get('early_exaggeration', 12),
                    init=params.get('init', 'random'),
                    random_state=params.get('random_state', 2018))

class MiniBatchDictLearning(DimReducerBase):
    def get_model(self, **params):
        return MiniBatchDictionaryLearning(n_components=params.get('n_components', 50),
                                           alpha=params.get('alpha', 1),
                                           batch_size=params.get('batch_size', 200),
                                           max_iter=params.get('max_iter', 25),
                                           random_state=params.get('random_state',2018))

    def run(self, x_train, x_validation=None):
        self.model.fit(x_train)
        x_train_analyzed = self.model.fit_transform(x_train)
        x_validation_analyzed = None
        if x_validation is not None and hasattr(self.model, 'transform'):
            x_validation_analyzed = self.model.transform(x_validation)
        
        return x_train_analyzed, x_validation_analyzed
    
class ICA(DimReducerBase):
    def get_model(self, **params):
        return FastICA(n_components=params.get('n_components', 25),
                       algorithm=params.get('algorithm', 'parallel'),
                       whiten=params.get('whiten', 'unit-variance'),
                       max_iter=params.get('max_iter', 100),
                       random_state=params.get('random_state', 2018))