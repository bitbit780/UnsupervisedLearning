import pandas as pd
import seaborn as sns
color = sns.color_palette()

from sklearn import preprocessing as pp

from sklearn.cluster import KMeans

import DataLoader
from DimReducer import PCAReducer

x_train, x_test, y_train, y_test, x_validation, y_validation = DataLoader.GetData('mnist')
train_index = range(0, len(x_train))

# アルゴリズムの登録
ALGO_MAP = {
    'PCA': PCAReducer,
    # 'Incremental PCA': IncrementalPCAReducer,
    # 'Sparse PCA': SparsePCAReducer,
    # 'Kernel PCA': KernelPCAReducer,
    # 'Singular Value Decompsition': SingularValueComosition,
    # 'Gaussian Random Projection': GaussianRandomProjectionReducer,
    # 'Sparse Random Projection': SparseRandomProjectionReducer,
    # 'Isomap': IsomapReducer,
    # 'Mulitidimentional Scaling': MultidimentionalScalingReducer,
    # 'Loccaly Linear Embedding': LLE,
    # 't-SNE':TSNEReducer,
    # 'Mini-batch Dictionary Learning': MiniBatchDictLearning,
    # 'Independent Component Analysis': ICA,
}

reducer = PCAReducer()
x_train_PCA, _ = reducer.run(x_train)
x_train_PCA = pd.DataFrame(data=x_train_PCA, index=train_index)
print(x_train_PCA)


n_clusters = 10
n_init = 10
max_iter = 300
tol = 0.0001
random_state = 2018

kMeans_inertia = pd.DataFrame(data=[], index=range(2,21), columns=['inertia'])

for n_clusters in range(2,21):
    kmeans = KMeans(n_clusters=n_clusters,
                    n_init=n_init,
                    max_iter=max_iter,
                    tol=tol,
                    random_state=random_state)
    cutoff = 99
    kmeans.fit(x_train_PCA.loc[:, 0:cutoff])
    kMeans_inertia.loc[n_clusters] = kmeans.inertia_
