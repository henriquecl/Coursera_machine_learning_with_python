import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# Importing the dataset

df = pd.read_csv('teleCust1000t.csv')
# Para utilizarmos a scikit-learn temos q converter de dataframe para Numpy Array, logo:
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender',
        'reside']].values
y = df['custcat'].values

# Normalize Data, nos retorna a meadia e a variancia
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Separando em dataset de treino e teste, utilizando a bib sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print(f'Train set: {X_train.shape}, {y_train.shape}')
print(f'Test set: {X_test.shape}, {y_test.shape}')

# Classification
# KNN

lista_train_set = []
lista_test_set = []

for k in range(1, 15):
    # Train model and predict
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    # Predicting
    y_hat = neigh.predict(X_test)
    # Acurracy
    train_set_acc = metrics.accuracy_score(y_train, neigh.predict(X_train))
    test_set_acc = metrics.accuracy_score(y_test, y_hat)
    train_set_acc_float = np.float(train_set_acc)
    test_set_acc_float = np.float(test_set_acc)
    lista_train_set.append(train_set_acc_float)
    lista_test_set.append(test_set_acc_float)

"""
print(f'Train set Accuracy: {train_set_acc}')
nump_to_int = np.int(train_set_acc)
print(type(nump_to_int))
"""
# Qual o K da a maior acurácia?
indice1 = lista_train_set.index(max(lista_train_set))
indice2 = lista_test_set.index(max(lista_test_set))
print(lista_test_set[indice2])
print(f'O K que fornece a maior acurácia no test set é : {indice2}')



