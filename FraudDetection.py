# Main
import numpy as np
import pandas as pd
import os

# Data Viz
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import matplotlib as mpl

# Data prep
from sklearn import preprocessing as pp
from scipy.stats import pearsonr
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

current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'credit_card_data', 'credit_card.csv'])
data = pd.read_csv(current_path + file)
# data.head()
# print(data.describe())
# print(data.columns)
# print("Number of fraudulent transactions:", data['Class'].sum())
# nonCounter = np.isnan(data).sum()
# print(nonCounter)
# distinctCounter = data.apply(lambda x: len(x.unique()))
# print(distinctCounter)

dataX = data.copy().drop(['Class'], axis=1)
dataY = data['Class'].copy()
featuresToScale = dataX.drop(['Time'], axis=1).columns
sX = pp.StandardScaler(copy=True)
dataX.loc[:,featuresToScale] = sX.fit_transform(dataX[featuresToScale])
print(dataX.describe())

# count_classes = data['Class'].value_counts(sort=True).sort_index()
# relative_freq = count_classes / len(data)
# plt.figure(figsize=(6,4))
# ax = sns.barplot(x=relative_freq.index, y=relative_freq.values)
# ax.set_title('Frequency Percentage by Class')
# ax.set_xlabel('Class')
# ax.set_ylabel('Frequency Percentage')
# plt.show()

# 訓練セットとテストセットに分割
x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=2018, stratify=dataY)
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)

# ロジスティック回帰
## ハイパーパラメータの設定
penalty = 'l2'
C = 1.0
class_weight = 'balanced'
random_state = 2018
solover = 'liblinear'
n_jobs = 1
logReg = LogisticRegression(penalty=penalty, C=C, class_weight=class_weight, random_state=random_state, solver=solover, n_jobs=n_jobs)

## モデルの訓練
trainingScores = []
cvScores = []
predictionsBaseOnKFolds = pd.DataFrame(data=[], index=y_train.index, columns=[0,1])
model = logReg

for train_index, cv_index in k_fold.split(np.zeros(len(x_train)), y_train.values.ravel()):
    x_train_fold, x_cv_fold = x_train.iloc[train_index,:], x_train.iloc[cv_index,:]
    y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
    model.fit(x_train_fold, y_train_fold)
    loglossTraining = log_loss(y_train_fold, model.predict_proba(x_train_fold)[:,1])
    trainingScores.append(loglossTraining)
    predictionsBaseOnKFolds.loc[x_cv_fold.index,:] = model.predict_proba(x_cv_fold)
    loglossCV = log_loss(y_cv_fold, predictionsBaseOnKFolds.loc[x_cv_fold.index,1])
    cvScores.append(loglossCV)
    print('Training Log Loss: ', loglossTraining)
    print('CV Log Loss: ', loglossCV)

loglossLogisticRegression = log_loss(y_train, predictionsBaseOnKFolds.loc[:,1])
print('Logistic Regression Log Loss:', loglossLogisticRegression)