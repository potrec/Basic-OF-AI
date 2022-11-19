import numpy as np
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC as SVM
from sklearn.tree import plot_tree


def tree_print(model, columns):
    # %% Rysowanie drzew
    import matplotlib.pyplot as plt
    plt.figure(figsize=(80, 40))
    plot_tree(model, feature_names=columns, filled=True, fontsize=35)
    plt.show()


def confusion_matrix_print(y_test, y_pred, X_train, y_train, X_test):
    # %% Macierz pomyłek
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    plt.show()


def kNN_SVM(X_train, X_test, y_train, y_test):
    # %% Testowanie kNN i SVM
    for weight in ('uniform', 'distance'):
        for neighbours in range(2, 10, 1):
            kNN_model = kNN(neighbours, weights=weight)
            kNN_model.fit(X_train, y_train)
            y_pred = kNN_model.predict(X_test)
            print("kNN results for: {} neighbours and '{}' weight function".format(neighbours, weight))
            print(f"precision: {precision_score(y_test, y_pred)}")
            print(f"specificity: {confusion_matrix(y_test, y_pred)[0, 0] / (confusion_matrix(y_test, y_pred)[0, 0] + confusion_matrix(y_test, y_pred)[0, 1])}")
            print("accuracy: {}".format(accuracy_score(y_test, y_pred)))
            print("f1 score: {}".format(f1_score(y_test, y_pred)))
            confusion_matrix_print(y_test, y_pred, X_train, y_train, X_test)

    for kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
        SVM_model = SVM(kernel=kernel)
        SVM_model.fit(X_train, y_train)
        y_pred = SVM_model.predict(X_test)
        print("SVM results for: {} kernel".format(kernel))
        print(f"precision: {precision_score(y_test, y_pred)}")
        print(f"specificity: {confusion_matrix(y_test, y_pred)[0, 0] / (confusion_matrix(y_test, y_pred)[0, 0] + confusion_matrix(y_test, y_pred)[0, 1])}")
        print("accuracy: {}".format(accuracy_score(y_test, y_pred)))
        print("f1 score: {}".format(f1_score(y_test, y_pred)))
        confusion_matrix_print(y_test, y_pred, X_train, y_train, X_test)


def DT(X_train, X_test, y_train, y_test, columns):
    # %% Testowanie DT
    from sklearn.tree import DecisionTreeClassifier as DT
    model = DT(max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("DT results for: {} max depth".format(5))
    print(f"precision: {precision_score(y_test, y_pred)}")
    print(f"specificity: {confusion_matrix(y_test, y_pred)[0, 0] / (confusion_matrix(y_test, y_pred)[0, 0] + confusion_matrix(y_test, y_pred)[0, 1])}")
    print("accuracy: {}".format(accuracy_score(y_test, y_pred)))
    print("f1 score: {}".format(f1_score(y_test, y_pred)))

    print(confusion_matrix(y_test, y_pred))
    confusion_matrix_print(y_test, y_pred, X_train, y_train, X_test)
    tree_print(model, columns)


def zad7():
    # %% Podział
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    columns = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022)
    # Skalowanie
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    kNN_SVM(X_train, X_test, y_train, y_test)
    DT(X_train, X_test, y_train, y_test, columns)


zad7()
