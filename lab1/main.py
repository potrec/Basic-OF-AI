import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

l_w = 1000
np.set_printoptions(linewidth=l_w)


def zad1():
    # Polecenie 1 Znajdź tablicę dwuwymiarową, która będzie wynikiem różnicy dwóch tablic: pierwsza
    # będzie zawierała wszystkie kolumny oraz parzyste wiersze tablicy danych
    # wejściowych, druga – wszystkie kolumny i nieparzyste wiersze tablicy danych
    data_csv = pd.read_csv("practice_lab_1.csv", sep=';')
    dane = pd.read_excel("practice_lab_1.xlsx")
    dane = dane.iloc[:100, :7]
    nazwy_kolumn = list(dane.columns)
    wartosciKolumn = np.array(dane.values)
    nazwyKolumny = np.array(nazwy_kolumn)
    arr1_1 = wartosciKolumn[::2, :] - wartosciKolumn[1::2, :]
    # Polecenie 2 Przekształć dane w sposób następujący: od każdej wartości odejmij średnią obliczoną
    # dla całej tablicy oraz podziel przez odchylenie standardowe wyznaczone dla całej
    # tablicy. Podpowiedź: skorzystaj z metod mean oraz std.
    arr1_2 = (wartosciKolumn - wartosciKolumn.mean()) / np.spacing(wartosciKolumn.std())
    print(
        f'Polecenie 2  \n Srednia wartość kolumn {wartosciKolumn.mean()} \n Wartość odchylenia standardowego kolumn {wartosciKolumn.std()}  \n {arr1_2} \n')
    # Polecenie 3 Wykonaj zadanie z podpunktu 2 dla oddzielnych kolumn pierwotnej tablicy, czyli
    # wyznaczając średnią oraz odchylenie standardowe dla oddzielnych kolumn.
    # Podpowiedź: aby uniknąć dzielenia przez zero do arr.std(axis=0) dodaj wynik funkcji
    # np.spacing(arr.std(axis=0)).
    arr1_3 = (wartosciKolumn - wartosciKolumn.mean(axis=0)) / (np.spacing(wartosciKolumn.std(axis=0)))
    print(
        f'Polecenie 3 \n Srednia wartość kolumn {wartosciKolumn.mean(axis=0)} \n Wartość odchylenia standardowego kolumn {np.spacing(wartosciKolumn.std(axis=0))} \n {arr1_3} \n')
    # Polecenie 4 Dla każdej kolumny pierwotnej tablicy policz współczynnik zmienności, definiowany
    # jako stosunek średniej do odchylenia standardowego, zabezpiecz się przed dzieleniem
    # przez 0 podobnie do poprzedniego punktu.
    mean = wartosciKolumn.mean(axis=0);
    std = np.spacing(wartosciKolumn.std(axis=0));
    print(f'Polecenie 4 \n')
    print(f'Srednia dla odczielnych kolumn {mean}\n Odchylenie standardowe dla odczielnych kolumn{std}');
    V = mean / std
    print(f'Współczynnik zmienności {V}')
    # 5. Znajdź kolumnę o największym współczynniku zmienności.
    print(f'Polecenie 5 \n')
    print(f'Kolumna o największym współczynniku zmienności {nazwy_kolumn[V.argmax()]}')

    # 6. Dla każdej kolumny pierwotnej tablicy policz liczbę elementów o wartości większej,
    # niż średnia tej kolumny.
    print(f'Polecenie 6  \n')
    print(f'Liczba elementów o wartości większej, niż średnia tej kolumny {np.sum(wartosciKolumn > mean, axis=0)}')

    # 7. Znajdź nazwy kolumn w których znajduje się wartość maksymalna. Podpowiedź: listę
    # stringów można również przekształcić na tablicę numpy, po czym można będzie dla
    # niej zastosować maskę.
    print(f'Polecenie 7 \n')
    kolumnaWartosciMaksymalnych = np.sum(wartosciKolumn == wartosciKolumn.max(), axis=0)
    mask7 = kolumnaWartosciMaksymalnych > 0
    result7 = nazwyKolumny[mask7]
    print(f'Nazwy kolumn w których jest wartość maksymalna{result7}')
    # 8. Znajdź nazwy kolumn w których jest najwięcej elementów o wartości 0. Podpowiedź:
    # wartości w tablicy wartości logicznych można sumować, zakładając, że zawiera ona
    # liczby całkowite, rzutowanie będzie wykonane automatycznie.
    print(f'Polecenie 8 \n')
    print(
        f'Nazwy kolumn w których jest najwięcej elementów o wartości 0: {nazwy_kolumn[np.argmax(np.sum(wartosciKolumn == 0, axis=0))]}')
    # 9. Znajdź nazwy kolumn w których suma elementów na pozycjach parzystych jest
    # większa od sumy elementów na pozycjach nieparzystych. Wyświetl ich nazwy,
    # postaraj się nie korzystać z pętli
    print(f'Polecenie 9 \n')
    even = wartosciKolumn[::2]
    odd = wartosciKolumn[1::2]
    mask = even.sum(axis=0) > odd.sum(axis=0)
    print(
        f'Nazwy kolumn w których suma elementów na pozycjach parzystych jest większa od sumy elementów na pozycjach nieparzystych: {nazwyKolumny[mask]}')


def zad2():
    # Wygeneruj wykresy ciągłe wartości następujących funkcji na przedziale [-5, 5]
    # z krokiem 0.01:
    # f(x) = tanh(x)
    # g(x) =(e^x - e^(-x))/(e^x + e^(-x))
    # h(x) = 1/(1 + e^(-x))
    # i(x) = x for x > 0, 0 for x <= 0
    # j(x) = x for x > 0, e^x - 1 for x <= 0

    x = np.arange(-5, 5, 0.01)
    f = np.tanh(x)
    g = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    h = 1 / (1 + np.exp(-x))
    i = np.where(x > 0, x, 0)
    j = np.where(x > 0, x, np.exp(x) - 1)

    plt.plot(x, f, label='f(x) = tanh(x)')
    plt.plot(x, g, label='g(x) =(e^x - e^(-x))/(e^x + e^(-x))')
    plt.plot(x, h, label='h(x) = 1/(1 + e^(-x))')
    plt.plot(x, i, label='i(x) = x for x > 0, 0 for x <= 0')
    plt.plot(x, j, label='j(x) = x for x > 0, e^x - 1 for x <= 0')

    plt.legend()
    plt.show()


def zad3():
    # Wczytaj dane z pliku do zmiennej typu pandas DataFrame. Za pomocą metody corr()
    # obiektu klasy DataFrame wygeneruj macierz korelacji, podstaw ją do oddzielnej zmiennej,
    # za pomocą okna podglądu odczytaj jej wartości.
    # Za pomocą modułu pyplot wygeneruj dwuwymiarową macierz wykresów punktowych
    # (scatter). Możesz skorzystać z pętli for. Zaobserwuj, jak wyglądają wykresy skorelowanych
    # oraz nieskorelowanych danych. Jak na podstawie wykresu można wyznaczyć znak
    # współczynnika korelacji?

    data_frame = pd.read_excel("practice_lab_1.xlsx", engine="openpyxl")
    corr_arr = data_frame.corr()

    data = data_frame.values

    fig, ax = plt.subplots(7, 7, figsize=(35, 35))
    x = np.arange(0, 100, 1)

    for i in range(7):
        for z in range(7):  # W celu uniknięcia powtórzeń można użyć range(i,7,1)
            y1 = data[:, i]
            y2 = data[:, z]
            ax[i, z].scatter(x, y1)
            ax[i, z].scatter(x, y2)
            ax[i, z].set_title("Kolumna {} vs Kolumna {}".format(i + 1, z + 1))
    plt.show()


zad1()
zad2()
zad3()
