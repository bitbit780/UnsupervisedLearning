#import libraries
## Main
import pandas as pd

## Data Viz
import seaborn as sns
color = sns.color_palette()

# Data Prep and Model Evaluation
from sklearn import preprocessing as pp
from scipy.stats import pearsonr
from numpy.testing import assert_array_almost_equal
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

# Algos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

import Utililty as util
import DataLoader

from DimReducer import *

x_train, x_test, y_train, y_test, x_validation, y_validation = DataLoader.GetData('mnist')
train_index = range(0, len(x_train))
test_index = range(len(x_train) + len(x_validation), len(x_train) + len(x_validation) + len(x_test))
validation_index = range(len(x_train), len(x_train) + len(x_validation))

# アルゴリズムの登録
ALGO_MAP = {
    'PCA': PCAReducer,
    'Incremental PCA': IncrementalPCAReducer,
    'Sparse PCA': SparsePCAReducer,
    'Kernel PCA': KernelPCAReducer,
    'Singular Value Decompsition': SingularValueComosition,
    'Gaussian Random Projection': GaussianRandomProjectionReducer,
    'Sparse Random Projection': SparseRandomProjectionReducer,
    'Isomap': IsomapReducer,
    'Mulitidimentional Scaling': MultidimentionalScalingReducer,
    'Loccaly Linear Embedding': LLE,
    't-SNE':TSNEReducer,
    'Mini-batch Dictionary Learning': MiniBatchDictLearning,
    'Independent Component Analysis': ICA,
}

algoName = 'Independent Component Analysis'

if algoName in ALGO_MAP:
    reducer = ALGO_MAP[algoName]()
    if algoName == 'Multidimentional Scaling':
        train_data = x_train.loc[0:1000, :]
        indices = train_index[0:1001]
    elif algoName == 'Kernel PCA':
        train_data = x_train.loc[:10000,:]
        indices = train_index[0:10001]
    elif algoName == 'Isomap':
        train_data = x_train.loc[0:5000,:]
        indices = train_index[0:5001]
    elif algoName == 'Mulitidimentional Scaling':
        train_data = x_train.loc[0:1000,:]
        indices = train_index[0:1001]
    elif algoName == 'Loccaly Linear Embedding':
        train_data = x_train.loc[0:5000,:]
        indices = train_index[0:5001]
    elif algoName == 't-SNE':
        train_data = x_train.loc[:5000,:9]
        indices = train_index[:5001]
    elif algoName == 'Mini-batch Dictionary Learning':
        train_data = x_train.loc[:,:10000]
        indices = train_index
    else:
        train_data = x_train
        indices = train_index

    x_train_analyzed_np, _ = reducer.run(train_data)
    x_train_analyzed = pd.DataFrame(data=x_train_analyzed_np, index=indices)

util.scatterPlot(x_train_analyzed, y_train, algoName)


