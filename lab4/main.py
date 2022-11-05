import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA, PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.metrics import confusion_matrix, accuracy_score

X, y = load_digits(return_X_y=True)

fig, ax = plt.subplots(1,10, figsize=(10,100))
for i in range(10):
    ax[i].imshow(X[i,:].reshape(8,8),cmap=plt.get_cmap('gray'))
ax[i].axis('off')
X.shape

