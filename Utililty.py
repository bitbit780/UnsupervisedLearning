import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()


def scatterPlot(xDF, yDF, alogName):
    tempDF = pd.DataFrame(data=xDF.loc[:,0:1], index=xDF.index)
    tempDF = pd.concat((tempDF, yDF), axis=1, join="inner")
    tempDF.columns = ["First Vector", "Second Vector", "Label"]
    sns.lmplot(x="First Vector", y="Second Vector", hue="Label", data=tempDF, fit_reg=False)
    ax = plt.gca()
    ax.set_title("Separation of Obsevations using " + alogName)
    plt.show()