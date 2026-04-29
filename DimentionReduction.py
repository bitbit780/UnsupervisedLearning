#import libraries
## Main
import numpy as np
import pandas as pd
import os, time
import pickle, gzip

## Data Viz
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import matplotlib as mpl

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

x_train, x_test, y_train, y_test, x_validation, y_validation = DataLoader.GetData('mnist')
train_index = range(0, len(x_train))
test_index = range(len(x_train) + len(x_validation), len(x_train) + len(x_validation) + len(x_test))
validation_index = range(len(x_train), len(x_train) + len(x_validation))

n_components = 784
whiten = False
random_state = 2018
batch_size = None

#algoName = 'PCA'
algoName = 'Incremental PCA'

# PCA
from sklearn.decomposition import PCA
# Incremental PCA
from sklearn.decomposition import IncrementalPCA

# PCA
if algoName == 'PCA':
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    x_train_analyzed = pca.fit_transform(x_train)
    x_train_analyzed = pd.DataFrame(data=x_train_analyzed, index=train_index)
# Incremental PCA
elif algoName == 'Incremental PCA':
    incrementalPCA = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    x_train_analyzed = incrementalPCA.fit_transform(x_train)
    x_train_analyzed = pd.DataFrame(x_train_analyzed, index=train_index)

    x_validation_analyzed = incrementalPCA.fit_transform(x_train)
    x_validation_analyzed = pd.DataFrame(x_train_analyzed, index=validation_index)    

util.scatterPlot(x_train_analyzed, y_train, algoName)






