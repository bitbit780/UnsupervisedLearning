import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()


def scatterPlot(xDF, yDF, algoName):
    tempDF = pd.DataFrame(data=xDF.loc[:,0:1], index=xDF.index)
    tempDF = pd.concat((tempDF, yDF), axis=1, join="inner")
    tempDF.columns = ["First Vector", "Second Vector", "Label"]
    sns.lmplot(x="First Vector", y="Second Vector", hue="Label", data=tempDF, fit_reg=False)
    ax = plt.gca()
    ax.set_title("Separation of Obsevations using " + algoName)
    plt.show()

def view_digit(example, x_train, y_train):
    label = y_train.loc[example]
    image = x_train.loc[example,:].values.reshape([28,28])
    plt.title('Example: %d Label: %d' % (example, label))
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()

def analyzeCluster(clusterDF, labelsDF):
    countByCluster = pd.DataFrame(data=clusterDF['cluster'].value_counts())
    countByCluster.reset_index(inplace=True, drop=False)
    countByCluster.columns = ['cluster', 'clusterCount']

    preds = pd.concat([labelsDF, clusterDF], axis=1)
    preds.columns = ['trueLabel', 'cluster']

    countByLabel = pd.DataFrame(data=preds.groupby('trueLabel').count())

    countMostFreq = pd.DataFrame(data=preds.groupby('cluster').agg(lambda x:x.value_counts().iloc[0]))
    countMostFreq.reset_index(inplace=True, drop=False)
    countMostFreq.columns = ['cluster', 'countMostFreaquent']

    accuracyDF = countMostFreq.merge(countByCluster, left_on="cluster", right_on="cluster")
    overallAccuracy = accuracyDF.countMostFreaquent.sum() / accuracyDF.clusterCount.sum()
    accuracyByLabel = accuracyDF.countMostFreaquent / accuracyDF.clusterCount

    return countByCluster, countByLabel, countMostFreq, accuracyDF, overallAccuracy, accuracyByLabel