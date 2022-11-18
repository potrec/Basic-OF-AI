import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
pd.options.mode.chained_assignment = None # żeby nie pojawiały się warningi z Pandas
TP = (0,0)  # True Positive
FN = (0,1)  # False Negative
FP = (1,0)  # False Positive
TN = (1,1)  # True Negative

data = pd.read_excel('practice_lab_3.xlsx')
columns = list(data.columns)
mask = data['Gender'].values == 'Female'
data.loc[mask, 'Gender'] = 1
data.loc[~mask, 'Gender'] = 0
cat_feature = pd.Categorical(data.Property_Area)
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])


def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data.loc[mask, column] = 1
    data.loc[~mask, column] = 0
    return data

def calculate_metrics(cm):
    print(cm[TN], cm[FP], cm[FN], cm[TP])
    se = cm[TP] / (cm[TP] + cm[FN])
    p = cm[TP] / (cm[TP] + cm[FP])
    sp = cm[TN] / (cm[TN] + cm[FP])
    acc = (cm[TP] + cm[TN]) / (cm[TP] + cm[TN] + cm[FP] + cm[FN])
    f1 = 2 * (p * se) / (p + se)
    return se, p, sp, acc, f1
def printMetrics(a_name, cm, se, p, sp, acc, f1):
    print(f"{a_name}\nMacierz:\n{cm}\nSensivity: {se}\nPrecision: {p}\nSpecificity: {sp}\nAccuracy: {acc}\nf1: {f1}\n")


def zad3(data):
    data = qualitative_to_0_1(data, 'Married', 'Yes')
    data = qualitative_to_0_1(data, 'Education', 'Graduate')
    data = qualitative_to_0_1(data, 'Self_Employed', 'Yes')
    data = qualitative_to_0_1(data, 'Loan_Status', 'Y')
    print(data)
def zad4():
    from sklearn.model_selection import train_test_split
    data = pd.read_excel('practice_lab_3.xlsx')
    columns = list(data.columns)
    mask = data['Gender'].values == 'Female'
    data.loc[mask, 'Gender'] = 1
    data.loc[~mask, 'Gender'] = 0
    cat_feature = pd.Categorical(data.Property_Area)
    one_hot = pd.get_dummies(cat_feature)
    data = pd.concat([data, one_hot], axis=1)
    data = data.drop(columns=['Property_Area'])
    col = list(data.columns)
    X = data.drop(columns=('Loan_Status')).values.astype(float)
    y = data['Loan_Status'].values.astype(float)

    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)
    from sklearn.neighbors import KNeighborsClassifier as kNN
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC as SVM
    models = [kNN(), SVM()]
    for model in models:
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        print(confusion_matrix(y_test, y_pred))

def zad5():
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

    # Z domyslnymi parametrami
    models = [kNN(), SVM()]
    cm_arr = []
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm_arr.append(confusion_matrix(y_test, y_pred))

    print("Domyslne parametry")
    se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
    printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

    # se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
    # printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)

    models = [kNN(n_neighbors=6, weights="uniform"), SVM(kernel="rbf")]
    cm_arr = []
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm_arr.append(confusion_matrix(y_test, y_pred))

    print("Z parametrami innymi niz domyslne")
    se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
    printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

    # se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
    # printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)

def zad6():
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

    models = [kNN(), SVM()]
    cm_arr = []
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm_arr.append(confusion_matrix(y_test, y_pred))

    print("Bez skalera")
    se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
    printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

    se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
    printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = [kNN(), SVM()]
    cm_arr = []
    for model in models:
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        cm_arr.append(confusion_matrix(y_test, y_pred))

    print("StandardScaler")
    se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
    printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

    se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
    printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)


    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = [kNN(), SVM()]
    cm_arr = []
    for model in models:
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        cm_arr.append(confusion_matrix(y_test, y_pred))

    print("MinMaxScaler")
    se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
    printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

    se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
    printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)


    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = [kNN(), SVM()]
    cm_arr = []
    for model in models:
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        cm_arr.append(confusion_matrix(y_test, y_pred))

    print("RobustScaler")
    se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
    printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

    se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
    printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)

def zad7():
    # Za pomocą kodu, który przedstawia Listing 3.7 załaduj dane, pozwalające
    # na wytrenowanie modelu do odróżnienia nowotworów łagodnych od złośliwych.
    # Przeprowadź analizę za pomocą omawianych metod klasyfikacji. Zbuduj drzewo decyzyjne o
    # wysokości 5, wygeneruj jego ilustrację, przedyskutuj, jakie cechy mają wpływ na wynik.
    # Porównaj wyniki z innymi metodami klasyfikacji.
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    # columns = list(data.columns)
    print(data)


# zad2(data)
# zad3(data)
# zad4()
zad5()
zad6()
# zad7()

