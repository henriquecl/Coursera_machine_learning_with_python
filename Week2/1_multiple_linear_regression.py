# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:20:48 2020

@author: HenriqueCampos
"""
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model

df = pd.read_csv('FuelConsumptionCo2.csv')
# Vamos escolher quais variáveis independentes vamo utilizar nesse caso.
var_dep = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY',
              'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Plotando o gráfico engine size x emissão
plt.scatter(var_dep['ENGINESIZE'], var_dep['CO2EMISSIONS'])
plt.xlabel('Engine Size')
plt.ylabel('C02 Emissions')
plt.show()

# Separando em Test e Training set
msk = np.random.rand(len(df)) < 0.8
train = var_dep[msk]
test = var_dep[~msk]

# Train data set distribuition
plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'])
plt.xlabel('Engine Size')
plt.ylabel('C02 Emission')
plt.show()

# Multiple Regression Model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])  # dep var
y = np.asanyarray(train[['CO2EMISSIONS']])  # ind var
regr.fit(x, y)
print(f'Os coeficientes encontrados foram: {regr.coef_}')

"""
-> Prediction: Ou seja, como a biblioteca vai encontrar os coeficientes da nossa regressão linear.
Como relatado nas aulas teóricas o tipo de predição utilizado pelo Scikit-learn é o Ordinary Least 
Squares method. Melhor para amostrar que possuem menos de 10.000 dados.  
"""
y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
rsq = np.mean((y_hat - y) ** 2)
print(f'Residual sum of squares: {np.around(rsq, decimals=3)}')
# Quanto mais próximo de 1 melhor.
print(f'Variance score: {np.around(regr.score(x, y), decimals=3)}')

