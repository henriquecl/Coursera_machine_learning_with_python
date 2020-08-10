import pandas as pd
import numpy as np
from sklearn import linear_model

# Importando o dataframe
df = pd.read_csv('FuelConsumptionCo2.csv')

# Separando em dataset de teste e de treino.
msk = np.random.random(len(df)) < 0.8
train = df[msk]
test = df[~msk]

# Fazendo a regressão
regr = linear_model.LinearRegression()
x_train = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
                              'FUELCONSUMPTION_HWY']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x_train, y_train)
print(f'Os coeficientes da regressão foram: {regr.coef_}')

# Encontrando o y chapéu
y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
                           'FUELCONSUMPTION_HWY']])
x_test = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
                             'FUELCONSUMPTION_HWY']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])
RSS = np.mean((y_hat - y_test) ** 2)
variance = regr.score(x_test, y_test)
print(f'O residual sum os squares foi: {RSS}, e a variance score foi: {variance}')