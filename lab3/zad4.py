# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None # żeby nie pojawiały się warningi z Pandas


TP = (0,0)  # True Positive
FN = (0,1)  # False Negative
FP = (1,0)  # False Positive
TN = (1,1)  # True Negative

def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data

def calculate_metrics(cm):
    print(cm[TP], cm[FN], cm[FP], cm[TN])
    sensivity = cm[TP] / (cm[TP] + cm[FN])
    precision = cm[TP] / (cm[TP] + cm[FP])
    specificity = cm[TN] / (cm[FP] + cm[TN])
    accuracy = (cm[TP] + cm[TN]) / (cm[TP] + cm[FN] + cm[FP] + cm[TN])
    f1 = (2 * sensivity * precision) / (sensivity + precision)

    return sensivity, precision, specificity, accuracy, f1

def printMetrics(a_name, cm, se, p, sp, acc, f1):
    print(f"{a_name}\nMacierz:\n{cm}\nSensivity: {se}\nPrecision: {p}\nSpecificity: {sp}\nAccuracy: {acc}\nf1: {f1}\n")

data = pd.read_csv("practice_lab_3.csv", sep=";")

# Przyporządkowanie binarnym wartosciom jakosciowym wartosci 0 lub 1
data = qualitative_to_0_1(data, 'Gender', 'Male')
data = qualitative_to_0_1(data, 'Married', 'Yes')
data = qualitative_to_0_1(data, 'Education', 'Graduate')
data = qualitative_to_0_1(data, 'Self_Employed', 'Yes')
data = qualitative_to_0_1(data, 'Loan_Status', 'Y')

# Przekształcenie nie binarnych wartoci jakosciowych na zbior cech o wartosciach 0 lub 1
# Utworzenie kodu 1-z-n dla wartosci jakosciowej
cat_feature = pd.Categorical(data['Property_Area'])
one_hot = pd.get_dummies(cat_feature)
# Zastapienie wartosci jakosciowej przez wygenerowany kod 1-z-n
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])


col = list(data.columns)
X = data.drop(columns=('Loan_Status')).values.astype(float)
y = data['Loan_Status'].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Z domyslnymi parametrami
models = [kNN(), SVM()]
cm_arr = []
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm_arr.append(confusion_matrix(y_test, y_pred))
print("Domyslne parametry")
se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
printMetrics("kNN", confusion_matrix(y_test, y_pred), se, p, sp, acc, f1)

se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
printMetrics("SVM", confusion_matrix(y_test, y_pred), se, p, sp, acc, f1)

# models = [kNN(n_neighbors=6, weights="uniform"), SVM(kernel="rbf")]
# cm_arr = []
# for model in models:
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     cm_arr.append(confusion_matrix(y_test, y_pred))
#
# print("Z parametrami innymi niz domyslne")
# se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
# printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)
#
# se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
# printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)

def zad4():
    def KNN():
        TN = 7
        FP = 26
        FN = 17
        TP = 73
        sensivity = TP / (TP + FN)
        precision = TP / (TP + FP)
        specificity = TN / (FP + TN)
        accuracy = (TP + TN) / (TP + FN + FP + TN)
        f1 = (2 * sensivity * precision) / (sensivity + precision)
        print(f"For kNN Sensivity: {sensivity}\nPrecision: {precision}\nSpecificity: {specificity}\nAccuracy: {accuracy}\nf1: {f1}\n")

    def SVM():
        TN = 0
        FP = 33
        FN = 0
        TP = 90
        sensivity = TP / (TP + FN)
        precision = TP / (TP + FP)
        specificity = TN / (FP + TN)
        accuracy = (TP + TN) / (TP + FN + FP + TN)
        f1 = (2 * sensivity * precision) / (sensivity + precision)
        print(f"For SVN Sensivity: {sensivity}\nPrecision: {precision}\nSpecificity: {specificity}\nAccuracy: {accuracy}\nf1: {f1}\n")

    KNN()
    SVM()

zad4()
