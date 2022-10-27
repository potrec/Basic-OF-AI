import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import seaborn as sns

l_w = 1000
np.set_printoptions(linewidth=l_w)
plt.rcParams['figure.figsize'] = [10, 5]


def printScatterPlot(x, y, xl="", yl="", title=""):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title(title)
    ax.set(xlabel=xl, ylabel=yl)
    plt.show()


def printLinearPlot(x, y, xl="", yl="", title=""):
    x = np.arange(-3, 3, 0.1).reshape((-1, 1))
    y = np.tanh(x) + np.random.randn(*x.shape) * 0.2
    ypred = LinearRegression().fit(x, y).predict(x)
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, ypred)
    plt.legend(['F(x) - aproksymująca',
                'f(x) - aproksymowana zaszumiona'])


def createRegresionPlot(X_train, y_train, X_test, y_test, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(X_train, y_train, color='blue')
    ax.scatter(X_test, y_test, color='red')
    ax.plot(X_test, y_pred, color='black', linewidth=3)
    plt.show()


def regression(X, y, n):
    arr = np.zeros(n)
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        linReg = LinearRegression()
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        arr[i] = mean_absolute_percentage_error(y_test, y_pred)
    return arr.mean()


def zad2_1():
    # Pobierz plik „practice_lab_2.xlsx” ze strony kursu. Podobnie do zadań z poprzednich zajęć
    # (Zadanie 1.4) wygeneruj macierz korelacji dla wczytanego zbioru. Przeanalizuj macierz
    # korelacji. Jakie zależności mogą mieć związek, a jakie są przypadkowe?
    # Wygeneruj wykresy korelacji pomiędzy cechami niezależnymi a cecha zależną (medianową ceną mieszkania)

    # Wczytanie danych
    data = pd.read_csv('practice_lab_2.csv', sep=';')
    col = data.columns.tolist()
    val = data.values
    # Wygeneruj macierz korelacji dla wczytanego zbioru.
    corr = data.corr()  # macierz korelacji
    print(corr)
    # wygenerowanie wykresów zależności mediany ceny mieszkań od poszczególnych wartości niezależnych
    for i in range(0, len(col) - 1):
        printScatterPlot(val[:, i], val[:, -1], xl=col[i], yl=col[-1], title=col[i] + " vs " + col[-1])
        printLinearPlot(val[:, i], val[:, -1], xl=col[i], yl=col[-1], title=col[i] + " vs " + col[-1])
        plt.show()


def zad2_2():
    data = pd.read_csv('practice_lab_2.csv', sep=';')
    col = data.columns.tolist()
    val = data.values
    # Podzielenie zbioriu na zbiór uczący i testowy
    X = val[:, :-1]
    y = val[:, -1]
    print(regression(X, y, 100))


def zad2_3():
    # Wykonaj Zadanie 2.2 dodając do niego procedurę usuwania/zastępowania wartości
    # odstających. Porównaj wyniki uzyskane w poprzednim zadaniu z nowymi wynikami.
    data = pd.read_csv('practice_lab_2.csv', sep=';')
    col = data.columns.tolist()
    val = data.values
    # Podzielenie zbioriu na zbiór uczący i testowy
    X = val[:, :-1]
    y = val[:, -1]

    # Funkcja usuwająca wartości odstające
    def removeOutliers(X_train, y_train):
        # Usuwanie wartości odstających
        outliers = np.abs((y_train - y_train.mean()) / y_train.std()) > 3  # wyznaczenie wartości odstajacych
        y_train_no_outliers = y_train[~outliers]  # usunięcie wartości odstających z zbioru wyjściowego
        X_train_no_outliers = X_train[~outliers, :]  # usunięcie wartości odstających z zbioru wejściowego
        return X_train_no_outliers, y_train_no_outliers

    # Funkcja zamieniająca wartości odstające na średnią
    def replaceOutliers(y_train):
        # Zamiana wartości odstających na średnią
        outliers = np.abs((y_train - y_train.mean()) / y_train.std()) > 3
        y_train_mean = y_train.copy()
        y_train_mean[outliers] = y_train.mean()
        return y_train_mean

    print(f'Wyniki przed procedurami: {regression(X, y, 100)}')
    replaceOutliers(y);
    print(f'Wyniki po procedurze zamiany wartości odstających na średnią: {regression(X, y, 100)}')
    removeOutliers(X, y);
    print(f'Wyniki po procedurze usunięcia wartości odstających: {regression(X, y, 100)}')


def zad2_4():
    # Spróbuj zaproponować cechy/kombinacje cech, które mogły by ulepszyć jakość
    # predykcji regresji liniowej.
    data = pd.read_csv('practice_lab_2.csv', sep=';')
    col = data.columns.tolist()
    val = data.values
    # Podzielenie zbioriu na zbiór uczący i testowy
    X = val[:, :-1]
    y = val[:, -1]
    # Generowanie nowych cech
    nowe_dane = np.stack([X[:, 4] / X[:, 7],
                          X[:, 4] / X[:, 5],
                          X[:, 4] * X[:, 3],
                          X[:, 4] / X[:, -1]], axis=-1)
    X_additional = np.concatenate([X, nowe_dane], axis=-1)
    print(f'Wyniki przed dodaniem nowych cech: {regression(X, y, 100)}')
    printLinearPlot(X[:, 4], X[:, 7], xl=col[4], yl=col[7], title=col[4] + " vs " + col[7])
    print(f'Wyniki po dodaniu nowych cech: {regression(X_additional, y, 100)}')
    printLinearPlot(X_additional[:, 4], X_additional[:, 7], xl=col[4], yl=col[7], title=col[4] + " vs " + col[7])


def zad2_5():
    from sklearn.datasets import load_diabetes
    import pandas as pd
    data = load_diabetes()
    dane=pd.DataFrame(data.data, columns=data.feature_names)
    dane['target']=data.target

    # Podzielenie zbioriu na zbiór uczący i testowy
    X = dane.iloc[:, :-1].values
    y = dane.iloc[:, -1].values

    print(f'Wyniki dla zbioru diabetes: {regression(X, y, 100)}')


# zad2_1()
# zad2_2()
# zad2_3()
# zad2_4()
zad2_5()

# %%
