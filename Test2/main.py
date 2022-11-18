import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier as DT, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

# #%% Ex1
data_frame = pd.read_csv('gender_classification.csv')
data_frame = pd.get_dummies(data_frame,columns=['Favorite Color','Favorite Music Genre','Favorite Beverage', 'Favorite Soft Drink'],
                            drop_first=True)
print(data_frame)

# #%% Ex2
X=data_frame.iloc[:,1:]
y=data_frame['Gender']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2022,stratify=y)
model_dt = DT(max_depth=3,random_state = 2022)
model_dt.fit(X_train,y_train)
y_pred = model_dt.predict(X_test)

plot_tree(model_dt,feature_names=data_frame.columns,class_names=['N','Y'])
plt.show()
print("Największy wpływ miały: ")
print(data_frame.columns[model_dt.feature_importances_.argmax()])
print(f" Czułość {accuracy_score(y_test,y_pred)}")
print(confusion_matrix(y_test, y_pred))

# #%% Ex3
print("Zadanie 3\n")

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

pipe1 = Pipeline([('scaler',MinMaxScaler()),('svm',SVC(kernel='poly'))])
pipe2 = Pipeline([('scaler',MinMaxScaler()),('knn',kNN(n_neighbors=7,weights='distance'))])

pipe1.fit(X_train,y_train)
pipe2.fit(X_train,y_train)

y_pred1 = pipe1.predict(X_test)
y_pred2 = pipe2.predict(X_test)

print(f"Accuracy SVM: {accuracy_score(y_test,y_pred1)}")
print(f"Accuracy KNN: {accuracy_score(y_test,y_pred2)}")

print("SVM lepiej zadziałał na zbiorze testowym\n")

# #%% Ex4
print("Zadanie 4\n")

data_frame = pd.read_csv('drift_database.csv')
X=data_frame.iloc[:,1:]
y=data_frame['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2022,stratify=y)

temp = PCA()
temp.fit(X_train)
variance = temp.explained_variance_ratio_
cumulated_variance = variance.cumsum()
num = (cumulated_variance<0.98).sum()+1

print(f"Liczba składowych głównych potrzebna do wyjaśnienia 98% wariancji: {num}")
