import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import confusion_matrix

def change_to_1_0(data, column, value):
    mask = data[column] == value
    data[column][mask] = 1
    data[column][~mask] = 0
    
    
def change_1_from_n(data, column):
    
    cat = pd.Categorical(data[column])
    hot = pd.get_dummies(cat)
    
    data = pd.concat([data, hot], axis=1)
    data = data.drop(column, axis=1)
    
    return data
    
    
data = pd.read_csv("drug200_cat.csv", sep=',')

#print(data)

data = change_1_from_n(data, 'BP')

X = data.drop('Drug', axis=1)
X = X.astype(float)

y = data['Drug']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2022)

model = DT(max_depth=3)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

matrix = confusion_matrix(y_test, y_pred)

print(matrix)
