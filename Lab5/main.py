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
        accs.append(accuracy_score(y_test_cv, y_pred))
    print(accs)
    print(np.mean(accs))

def zadanie3():
    # Napisz skrypt, który pozwoli na dokonanie wyszukiwania siatkowego w celu
    # znalezienia najbardziej obiecujących wartości hiperparametrów. Przetestuj takie parametry,
    # jako liczba warstw, liczba neuronów w warstwie, funkcja aktywacji, optymalizator, prędkość
    # nauczania. Uwzględnij zjawisko przetrenowania – oznacza to, że najlepszy wynik
    # nie koniecznie będzie po ostatniej epoce uczenia sieci.
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    data = load_digits()
    X = data.data
    y = data.target
    y = pd.Categorical(y)
    y = pd.get_dummies(y).values
    class_num = y.shape[1]


    from keras.wrappers.scikit_learn import KerasClassifier
    from scipy.stats import reciprocal
    from keras.models import Sequential
    from keras.layers import Input, Dense
    from keras.optimizers import Adam, RMSprop, SGD
    from keras.utils import plot_model
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

    def build_model(layers, activation, optimizer, learning_rate):
        model = Sequential()
        model.add(Dense(layers[0], input_shape = (X.shape[1],), activation = activation))
        for i in range(1, len(layers)):
            model.add(Dense(layers[i], activation = activation))
        model.add(Dense(class_num, activation = 'softmax'))
        model.compile(optimizer= optimizer(learning_rate),
                            loss='categorical_crossentropy',
                            metrics=('accuracy'))
        return model

    model = KerasClassifier(build_fn=build_model, verbose=0) # verbose=0 - nie wyswietla informacji o epokach
    # parametry do wyszukiwania
    # Liczba warstw
    layers = [[64], [128], [64, 64], [128, 128]]
    # Liczb neuronow w warstwie
    neurons = [16, 32, 64, 128]
    # Funkcja aktywacji
    activation = ['relu', 'tanh', 'sigmoid']
    # Optymalizator
    optimizer = [Adam, RMSprop, SGD]
    # Prędkość nauczania
    learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    # parametry do wyszukiwania
    param_grid = dict(layers=layers, activation=activation, optimizer=optimizer, learning_rate=learning_rate, batch_size=[32, 64], epochs=[10])
    # wyszukiwanie siatkowe
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X, y)
    # najlepsze parametry
    print("Najlepsze parametry: %s z wynikiem %f" % (grid_result.best_params_, grid_result.best_score_))

    # layers = [[64,64,64], [128,128,128], [256,256,256]]  # liczba neuronow w warstwach
    # activations = ['relu', 'tanh'] # funkcje aktywacji
    # optimizers = [Adam, RMSprop, SGD] # optymalizatory
    # learning_rates = [0.001, 0.01, 0.1] # predkosci nauczania
    # param_grid = dict(layers=layers, activation=activations, optimizer=optimizers, learning_rate=learning_rates)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # grid_result = grid.fit(X_train, y_train)
    # print(grid_result.best_score_)
    # print(grid_result.best_params_)
    # print(grid_result.best_estimator_)



    # model = KerasClassifier(build_fn=build_model, verbose=0)
    # param_grid = {
    #     'n_hidden': [[64,64,64], [128,128,128], [64,64,64,64], [128,128,128,128]],
    #     'n_neurons': [32, 64, 128, 256],
    #     'activation': ['relu', 'tanh', 'sigmoid'],
    #     'optimizer': [Adam, RMSprop, SGD],
    #     'learning_rate': reciprocal(1e-4, 1e-2),
    #     'batch_size': [32, 64, 128, 256],
    #     'epochs': [100, 200, 300, 400]\
    # }
    # grid = RandomizedSearchCV(model, param_grid, cv=3, n_iter=10, verbose=2)
    # grid.fit(X, y)
    # print(grid.best_params_)
    # # grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    # grid_result = grid.fit(X, y)
    #
    # print('Best score: ', grid_result.best_score_)
    # print('Best params: ', grid_result.best_params_)
    # print('Best estimator: ', grid_result.best_estimator_)
    # print('Best index: ', grid_result.best_index_)




    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=2022)
    # scaler=StandardScaler()
    # X_train=scaler.fit_transform(X_train)
    # X_test=scaler.transform(X_test)
    # num_epochs=10
    #
    # def build_model(n_hidden, n_neurons, learning_rate, activation,optimizer):
    #     model = Sequential()
    #     model.add(Dense(n_neurons, input_shape = (X.shape[1],), activation = activation))
    #     for layer in range(n_hidden):
    #         model.add(Dense(n_neurons, activation=activation))
    #     model.add(Dense(class_num, activation='softmax'))
    #     model.compile(optimizer=optimizer(learning_rate), loss='categorical_crossentropy', metrics=('accuracy'))
    #     return model
    # #%%
    # # RANDOMIZED SEARCH
    # keras_classifier=KerasClassifier(build_model)
    # # w randomized search mozna dac wiecej kombinacji bo sa sprawdzane losowe z nich
    # param_distribs={
    #     'n_hidden': [0,1,2,3],
    #     'n_neurons': np.arange(1,100),
    #     'learning_rate': reciprocal(3e-4, 3e-2),
    #     'activation': ['relu','selu','softmax'],
    #     'optimizer': [SGD,Adam,RMSprop]
    # }
    # rnd_search_cv=RandomizedSearchCV(keras_classifier, param_distribs, n_iter=10, cv=5)
    # rnd_search_cv.fit(X_train, y_train, epochs=num_epochs)
    #
    # best_params_from_random=rnd_search_cv.best_params_
    # best_model_from_random=rnd_search_cv.best_estimator_
    #
    # print(best_params_from_random,best_model_from_random)





zadanie2()
# zadanie3()

