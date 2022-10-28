import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

#Zadanie 1
data_frame = pd.read_csv("Concrete_Data_Out.csv")
data = data_frame.values
data_columns = np.array(data_frame.columns)
print(data)
print(data_columns)

# Podziel zbiór na X i y. W zbiorze X znajdź nazwę kolumny,
# która zawiera najwięcej wartości zerowych.

X = data[:,:-1]
y = data[:,-1]

# Znajdź kolumnę, która zawiera najwięcej wartości zerowych.Wypisz nazwę tej kolumny.
zeros = np.count_nonzero(X == 0, axis=0)
print(f' Koluman która zawiera najwięcej wartości zerowych to: {data_columns[np.argmax(zeros)]}')


#Zadanie 2
#Podziel cały zbiór na uczący i testowy w proporcji 7:3. Ustaw random_state=2022, shuffle=False.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022, shuffle=False)

# Sprawdź, czy w zbiorze znajdują się wartości odstające. Przyjmij, że wartości
# odstające to takie, które nie mieszczą się w przedziale (srednia - 3*odchylenie standardowe, srednia + 3*odchylenie standardowe).
# Wyświetl liczbę wartości odstających.
def check_outliers(y_train):
    mean = np.mean(y_train)
    std = np.std(y_train)
    outliers = 0
    for i in y_train:
        if i < mean - 3*std or i > mean + 3*std:
            outliers += 1
    return outliers
print(f' Liczba wartości odstających w zbiorze uczącym: {check_outliers(y_train)}')

def remove_outliers(X_train, y_train):
    mean = np.mean(y_train)
    std = np.std(y_train)
    outliers = []
    for i in range(len(y_train)):
        if y_train[i] < mean - 3*std or y_train[i] > mean + 3*std:
            outliers.append(i)
    return np.delete(X_train, outliers, axis=0), np.delete(y_train, outliers, axis=0)
x_train, y_train = remove_outliers(X_train, y_train)


# 3. Na zbiorze uczącym wytrenuj model regresji. Wygeneruj wykres słupkowy (bar) dla
# współczynników (wag) modelu. Na jego podstawie napisz w komentarzu, która cecha ma
# największy wpływ na wytrzymałość cementu.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2022,shuffle=False)
linReg = LinearRegression()
linReg.fit(X_train,y_train)
weights = linReg.coef_
weights_names = data_columns[:-1]
plt.bar(weights_names, weights)
plt.show()
print(f' Największy wpływ na wytrzymałość cementu ma cecha: {weights_names[np.argmax(weights)]}\n')

#zadanie 4
# Dokonaj predykcji z modelu na zbiorze testowym. Oblicz i wyświetl wartości trzech znanych
# Ci miar jakości pracy modelu.
y_pred = linReg.predict(X_test)
print(f' MAPE: {mean_absolute_percentage_error(y_test, y_pred)}')
print(f' MSE: {mean_squared_error(y_test, y_pred)}')
print(f' MAE: {mean_absolute_error(y_test, y_pred)}')


