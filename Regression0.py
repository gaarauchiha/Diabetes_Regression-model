# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 04:25:18 2023

@author: Reza Gouklaney
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #To get a 3D plot
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# load data using pandas
data_df = pd.read_csv('train.csv')
data_df.head()


y_df = data_df['y']
X_df = data_df.drop(['y'], axis=1)
X_df.head()

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_df['Col 1'].values, X_df['Col 2'].values, y_df.values)
ax.set_xlabel('Col 1')
ax.set_ylabel('Col 2')
ax.set_zlabel('y')

X = X_df.values
y = y_df.values


# extend one column for the bias term
X = np.hstack((np.ones((X.shape[0],1)),X))
# print (X)
# Use matrix calculation to solve
w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

predicted_y = np.dot(X, w)

# training error
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y, predicted_y))