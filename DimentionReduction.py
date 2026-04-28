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
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
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

# Load the datasets
current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'mnist_data', 'mnist.pkl.gz'])

f = gzip.open(current_path+file, 'rb')
train_set, validation_set, test_set = pickle.load(f, encoding='latin1')
f.close()

x_train, y_train = train_set[0], train_set[1]
x_validation, y_validation = validation_set[0], validation_set[1]
x_test, y_test = test_set[0], test_set[1]

# Verify shape of datasets
# print("Shape of x_train: ", x_train.shape)
# print("Shape of y_train: ", y_train.shape)
# print("Shape of x_validation", x_validation.shape)
# print("Shape of y_validation", y_validation.shape)
# print("Shape of x_test", x_test.shape)
# print("Shape of y_test", y_test.shape)

# Create Pands DataFrames from the datasets
train_index = range(0, len(x_train))
validation_index = range(len(x_train), len(x_train) + len(x_validation))
test_index = range(len(x_train) + len(x_validation), len(x_train) + len(x_validation) + len(x_test))

x_train = pd.DataFrame(data=x_train, index=train_index)
y_train = pd.Series(data=y_train, index=train_index)

x_validation = pd.DataFrame(data=x_validation, index=validation_index)
y_validation = pd.Series(data=y_validation, index=validation_index)

x_test = pd.DataFrame(data=x_test, index=test_index)
y_test = pd.Series(data=y_test, index=test_index)

# Describe the training matrix
# print(x_train.describe())

# Show the labels
# print(y_train.head())

def view_digit(example):
    label = y_train.loc[example]
    image = x_train.loc[example,:].values.reshape([28,28])
    plt.title('Example: %d Label: %d' % (example, label))
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()

n_components = 784
whiten = False
random_state = 2018

# pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)

# x_train_PCA = pca.fit_transform(x_train)
# x_train_PCA = pd.DataFrame(data=x_train_PCA, index=train_index)

# Percentage of Variance Captured by 784 principal components
# print("Variance Explained by all 784 principal componets: ",sum(pca.explained_variance_ratio_))

# Percentage of Variance Captured by X principal components
# importanceOfPrincepalComponents = pd.DataFrame(data=pca.explained_variance_ratio_)
# importanceOfPrincepalComponents = importanceOfPrincepalComponents.T

# print('Variance Captured by First 10 Princepal Components: ', importanceOfPrincepalComponents.loc[:, 0:9].sum(axis=1).values)
# print('Variance Captured by First 20 Princepal Components: ', importanceOfPrincepalComponents.loc[:, 0:19].sum(axis=1).values)
# print('Variance Captured by First 50 Princepal Components: ', importanceOfPrincepalComponents.loc[:, 0:49].sum(axis=1).values)
# print('Variance Captured by First 100 Princepal Components: ', importanceOfPrincepalComponents.loc[:, 0:99].sum(axis=1).values)
# print('Variance Captured by First 200 Princepal Components: ', importanceOfPrincepalComponents.loc[:, 0:199].sum(axis=1).values)
# print('Variance Captured by First 300 Princepal Components: ', importanceOfPrincepalComponents.loc[:, 0:299].sum(axis=1).values)

batch_size = None
incrementalPCA = IncrementalPCA(n_components=n_components, batch_size=batch_size)
x_train_incrementalPCA = incrementalPCA.fit_transform(x_train)
x_train_incrementalPCA = pd.DataFrame(x_train_incrementalPCA, index=train_index)

x_validation_incrementalPCA = incrementalPCA.fit_transform(x_train)
x_validation_incrementalPCA = pd.DataFrame(x_train_incrementalPCA, index=validation_index)

util.scatterPlot(x_train_incrementalPCA, y_train, "Incremental PCA")
# util.scatterPlot(x_train_PCA, y_train, "PCA")


