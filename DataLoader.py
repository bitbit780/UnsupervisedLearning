import os
import pandas as pd
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
import pickle, gzip

def GetData(dataName):
    current_path = os.getcwd()
    if dataName == 'credit_card':
        file = os.path.sep.join(['', 'datasets', 'credit_card_data', 'credit_card.csv'])
        data = pd.read_csv(current_path + file)
        dataX = data.copy().drop(['Class'], axis=1)
        dataY = data['Class'].copy()
        featuresToScale = dataX.drop(['Time'], axis=1).columns
        sX = pp.StandardScaler(copy=True)
        dataX.loc[:,featuresToScale] = sX.fit_transform(dataX[featuresToScale])
        # 訓練セットとテストセットに分割
        x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=2018, stratify=dataY)
        x_validation, y_validation = None, None
    elif dataName == 'mnist':
        file = os.path.sep.join(['', 'datasets', 'mnist_data', 'mnist.pkl.gz'])
        f = gzip.open(current_path+file, 'rb')
        train_set, validation_set, test_set = pickle.load(f, encoding='latin1')
        f.close()
        x_train, y_train = train_set[0], train_set[1]
        x_validation, y_validation = validation_set[0], validation_set[1]
        x_test, y_test = test_set[0], test_set[1]

        train_index = range(0, len(x_train))
        validation_index = range(len(x_train), len(x_train) + len(x_validation))
        test_index = range(len(x_train) + len(x_validation), len(x_train) + len(x_validation) + len(x_test))

        x_train = pd.DataFrame(data=x_train, index=train_index)
        y_train = pd.Series(data=y_train, index=train_index)

        x_validation = pd.DataFrame(data=x_validation, index=validation_index)
        y_validation = pd.Series(data=y_validation, index=validation_index)

        x_test = pd.DataFrame(data=x_test, index=test_index)
        y_test = pd.Series(data=y_test, index=test_index)

    return x_train, x_test, y_train, y_test, x_validation, y_validation






