import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 16
np.set_printoptions(precision=2)

def listing():
    # x = np.arange(0,1,0.01)
    # y = x.copy()
    # X,Y = np.meshgrid(x,y)
    # wx = 0.1
    # wy = 0.3
    # S = wx*X+wy*Y
    # out = S>0.15
    # fig, ax = plt.subplots(1,1)
    # ax.imshow(out)
    # ticks = np.around(np.arange(-0.2,1.1,0.2), 3)
    # ax.set_xticklabels(ticks)
    # ax.set_yticklabels(ticks)
    # plt.gca().invert_yaxis()
    #%% zaladowanie danych

    data = load_iris()
    y = data.target
    X = data.data
    y = pd.Categorical(y)
    y = pd.get_dummies(y).values
    class_num = y.shape[1]

    #%% tworzenie sieci sekwenyjnego modelu sieci neuronowej

    model = Sequential()
    model.add(Dense(64, input_shape = (X.shape[1],), activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(class_num, activation = 'softmax'))
    learning_rate = 0.0001
    model.compile(optimizer= Adam(learning_rate),
                     loss='categorical_crossentropy',
                     metrics=('accuracy'))
    model.summary()
    plot_model(model,to_file="my_model.png")
    #%% trenowanie sieci

    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                          test_size = 0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model.fit(X_train, y_train, batch_size=32, epochs=100,
                validation_data=(X_test, y_test), verbose=2)

    #%% przetrenoowanie sieci

    historia = model.history.history
    floss_train = historia['loss']
    floss_test = historia['val_loss']
    acc_train = historia['accuracy']
    acc_test = historia['val_accuracy']
    fig,ax = plt.subplots(1,2, figsize=(20,10))
    epochs = np.arange(0, 100)
    ax[0].plot(epochs, floss_train, label = 'floss_train')
    ax[0].plot(epochs, floss_test, label = 'floss_test')
    ax[0].set_title('Funkcje strat')
    ax[0].legend()
    ax[1].set_title('Dokladnosci')
    ax[1].plot(epochs, acc_train, label = 'acc_train')
    ax[1].plot(epochs, acc_test, label = 'acc_test')
    ax[1].legend()
    plt.show()

    #%% walidacja krzyzowa
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                          test_size=0.2)
    accs = []
    scaler = StandardScaler()
    for train_index, test_index in KFold(5).split(X_train):
                        X_train_cv = X_train[train_index,:]
    X_test_cv = X_train[test_index,:]
    y_train_cv = y_train[train_index,:]
    y_test_cv = y_train[test_index,:]
    X_train_cv = scaler.fit_transform(X_train_cv)
    X_test_cv = scaler.transform(X_test_cv)
    model.fit(X_train, y_train, batch_size=32,
              epochs=100, validation_data=(X_test, y_test),
              verbose=2)

def zadanie2():
    # Samodzielnie zaimplementuj proces uczenia sieci neuronowej. Wygeneruj wykresy
    # metryk oraz funkcji strat dla zbioru treningowego oraz testowego. Jako dane wejściowe
    # wykorzystaj zbiór danych MNIST, którego ładowanie pokazuje Listing 5.8

    #%% zaladowanie danych
    from sklearn.datasets import load_digits
    data = load_digits()
    X = data.data
    y = data.target
    y = pd.Categorical(y)
    y = pd.get_dummies(y).values
    class_num = y.shape[1]

    #%% tworzenie sieci sekwenyjnego modelu sieci neuronowej
    from keras.models import Sequential
    from keras.layers import Input, Dense
    from keras.optimizers import Adam, RMSprop, SGD
    from keras.utils import plot_model
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_iris
    from matplotlib import pyplot as plt
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score

    model = Sequential()
    model.add(Dense(64, input_shape = (X.shape[1],), activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(class_num, activation = 'softmax'))
    learning_rate = 0.0001
    model.compile(optimizer= Adam(learning_rate),
                        loss='categorical_crossentropy',
                        metrics=('accuracy'))
    model.summary()
    plot_model(model,to_file="my_model.png")
    #%% trenowanie sieci
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), verbose=2)
    #%#  generowanie wykresow metryk oraz funkcji strat dla zbioru treningowego oraz testowego
    historia = model.history.history
    floss_train = historia['loss']
    floss_test = historia['val_loss']
    acc_train = historia['accuracy']

    acc_test = historia['val_accuracy']
    fig,ax = plt.subplots(1,2, figsize=(20,10))
    epochs = np.arange(0, 100)
    ax[0].plot(epochs, floss_train, label = 'floss_train')
    ax[0].plot(epochs, floss_test, label = 'floss_test')
    ax[0].set_title('Funkcje strat')
    ax[0].legend()
    ax[1].set_title('Dokladnosci')
    ax[1].plot(epochs, acc_train, label = 'acc_train')
    ax[1].plot(epochs, acc_test, label = 'acc_test')
    ax[1].legend()
    plt.show()
#%% walidacja krzyzowa
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    accs = []
    scaler = StandardScaler()
    #przetestuj siec z parametry jak liczba warstw, liczba neuronow, funkcje aktywacji, optymalizator, prędkość uczenia
    # krotka z liczbą warstw, liczbą neuronów, funkcją aktywacji, optymalizatorem, prędkością uczenia
    # dla każdego zestawu parametrów wykonaj walidację krzyżową
    # dla każdego zestawu parametrów wyznacz średnią dokładność
    # dla każdego zestawu parametrów wyznacz odchylenie standardowe dokładności
    # wybierz zestaw parametrów, który daje najlepszą średnią dokładność


    for train_index, test_index in KFold(5).split(X_train):
        X_train_cv = X_train[train_index,:]
        X_test_cv = X_train[test_index,:]
        y_train_cv = y_train[train_index,:]
        y_test_cv = y_train[test_index,:]
        X_train_cv = scaler.fit_transform(X_train_cv)
        X_test_cv = scaler.transform(X_test_cv)
        model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), verbose=2)
        y_pred = model.predict(X_test_cv)
        y_pred = np.argmax(y_pred, axis=1)
        y_test_cv = np.argmax(y_test_cv, axis=1)
        acc = accuracy_score(y_test_cv, y_pred)
        accs.append(acc)
    print(accs)
    print(np.mean(accs))






zadanie2()
