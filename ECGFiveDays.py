import DataLoader
import numpy as np

from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

x_train, x_test, y_train, y_test, _, _, data_train, data_test = DataLoader.GetData("ECG5Days")
# print("Number of time series:", len(data_train))
# print("Number of unique classes:", len(np.unique(data_train[:,0])))
# print("Time series length:", len(data_train[0,1:]))

print(data_train[:,0])
# # Examples of Class 1.0
# for i in range(0,10):
#     if data_train[i, 0] == 1.0:
#         print("Plot ", i, " Class ", data_train[i,0])
#         plt.plot(data_train[i])
#         plt.show()

# # Examples of Class 1.0
# for i in range(0,10):
#     if data_train[i, 0] == 2.0:
#         print("Plot ", i, " Class ", data_train[i,0])
#         plt.plot(data_train[i])
#         plt.show()

#Prepare the data - Scale
x_train = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(x_train)
x_test = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(x_test)

# #Train using k-Shape
ks = KShape(n_clusters=2, max_iter=100, n_init=100, verbose=0)
ks.fit(x_train)

# Make predictions and calculate adjusted Rand index
preds = ks.predict(x_train)
print(preds)
ars = adjusted_rand_score(data_train[:,0], preds)
print("Adjusted Rand Index:", ars)

# Make predictions on test set and cluculate adusted Rand index
preds_test = ks.predict(x_test)
ars = adjusted_rand_score(data_test[:,0], preds_test)
print("Adjusted Rand Index:", ars)