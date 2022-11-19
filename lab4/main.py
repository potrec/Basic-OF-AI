from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA, PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.pipeline import Pipeline


def listing():
    X, y = load_digits(return_X_y=True)
    fig, ax = plt.subplots(1, 10, figsize=(10, 100))
    for i in range(10):
        ax[i].imshow(X[i, :].reshape(8, 8), cmap=plt.get_cmap('gray'))
    ax[i].axis('off')
    X.shape
    plt.show()
    # %%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022, stratify=y)
    pca_transform = PCA()
    pca_transform.fit(X_train)
    variances = pca_transform.explained_variance_ratio_
    cumulated_variances = variances.cumsum()
    plt.scatter(np.arange(variances.shape[0]), cumulated_variances)
    plt.show()
    # %%

    PC_num = (cumulated_variances < 0.95).sum() + 1  # wazne zeby dodac 1, poniewaz␣chcemy zeby 95% bylo PRZEKROCZONE
    print("Aby wyjaśnić 95% wariancji, potrzeba " + str(PC_num) + ' składowych głównych')
    pca95 = PCA(n_components=0.95)
    X_train_pca = pca95.fit_transform(X_train)
    pca95.n_components_
    X_test_pca = pca95.transform(X_test)
    # %%
    scaler = StandardScaler()
    X_train_pca_scaled = scaler.fit_transform(X_train_pca)
    X_test_pca_scaled = scaler.transform(X_test_pca)
    model = kNN(n_neighbors=5, weights='distance')
    model.fit(X_train_pca_scaled, y_train)
    y_predict = model.predict(X_test_pca_scaled)
    print(confusion_matrix(y_test, y_predict))
    print(accuracy_score(y_test, y_predict))
    # %%
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ['transformer', PCA(0.95)],
        ['scaler', StandardScaler()],
        ['classifier', kNN(weights='distance')]
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    # %%
    import seaborn as sns
    aux = X_train_pca.copy()
    aux = pd.DataFrame(aux)
    aux.columns = ['PC' + str(i + 1) for i in range(28)]
    aux['target'] = y_train
    sns.lmplot(x='PC1', y='PC2', hue='target', data=aux, fit_reg=False)
    plt.show()
    # %%


def pd():
    import pandas as pd
    df = pd.read_csv('ionosphere_data.csv', header=None)
    df.columns = ["C" + str(i) for i in range(36)]
    df.drop("C0", axis=1, inplace=True)
    df.shape
    df.iloc[:, -1].value_counts()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022, stratify=y)
    pca_transform = PCA()
    pca_transform.fit(X_train)
    variances = pca_transform.explained_variance_ratio_
    cumulated_variances = variances.cumsum()
    plt.scatter(np.arange(variances.shape[0]), cumulated_variances)
    plt.show()
    PC_num = (cumulated_variances < 0.95).sum() + 1
    print("Aby wyjaśnić 95% wariancji, potrzeba " + str(PC_num) + ' składowych głównych \n')
    pca95 = PCA(n_components=0.95)
    pca95.fit(X_train)

    # krotka zawierajaca metody do redukcji wymiarowosci
    methods = [PCA(PC_num), FastICA(PC_num, random_state=2022)]
    # krotka zawierajaca metody do skalowania
    scalers = [StandardScaler(), MinMaxScaler(), RobustScaler(), None]
    # krotka zawierajaca metody klasyfikacji
    classifiers = [kNN(n_neighbors=5, weights='distance'), SVC(), DT(max_depth=5), RandomForest(max_depth=5)]
    # krotka zawierajaca nazwy metod
    names = ['PCA', 'ICA']
    # krotka zawierajaca nazwy skalowania
    scalers_names = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'None']
    # krotka zawierajaca nazwy klasyfikatorow
    classifiers_names = ['kNN', 'SVC', 'DT', 'RandomForest']
    # macierz zawierajaca wyniki
    results = np.zeros((len(names), len(scalers_names), len(classifiers_names)))
    # petla po metodach redukcji wymiarowosci
    for i in range(len(methods)):
        # petla po skalowaniach
        for j in range(len(scalers)):
            # petla po klasyfikatorach
            for k in range(len(classifiers)):
                # %%tworzenie pipeline
                if scalers[j] is None:
                    pipe = Pipeline([
                        ['transformer', methods[i]],
                        ['classifier', classifiers[k]]
                    ])
                else:
                    pipe = Pipeline([
                        ['transformer', methods[i]],
                        ['scaler', scalers[j]],
                        ['classifier', classifiers[k]]
                    ])

                # uczenie modelu
                pipe.fit(X_train, y_train)
                # predykcja
                y_pred = pipe.predict(X_test)

                # %%obliczanie wynikow
                results[i, j, k] = accuracy_score(y_test, y_pred)
                print("Accuracy dla " + names[i] + " + " + scalers_names[j] + " + " + classifiers_names[
                    k] + " wynosi " + str(results[i, j, k]))

    # %% wyswietlam ktora kombinacja daje najlepszy wynik
    print("\nNajlepszy wynik to " + str(results.max()) + " dla " + names[
        np.unravel_index(results.argmax(), results.shape)[0]] + " + " + scalers_names[
              np.unravel_index(results.argmax(), results.shape)[1]] + " + " + classifiers_names[
              np.unravel_index(results.argmax(), results.shape)[2]])

pd()
