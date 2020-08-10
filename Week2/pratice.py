import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pylab as pl

# Importando o dataframe
df = pd.read_csv('FuelConsumptionCo2.csv')
ind_var = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY',
              'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Vendo o gráfico.
plt.scatter(ind_var['FUELCONSUMPTION_CITY'], ind_var['CO2EMISSIONS'])
plt.xlabel('Consumo na Cidade')
plt.ylabel('CO2 Emissions')
plt.show()

# Criando o test set e training set
msk = np.random.rand(len(df)) < 0.8  # Pega linhas aleatórias de df
train = df[msk]
test = df[~msk]

# Plotando gráfico do Train data   - Não tem a ver com a regressão em si.
plt.scatter(train['FUELCONSUMPTION_CITY'], train['CO2EMISSIONS'])
plt.xlabel('Consumo na Cidade')
plt.ylabel('Emissão de CO2')
plt.show()

# Fazendo a regressão linear
from sklearn import linear_model

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
# Vars independentes q qremos utilizar como
# parametro da regressão
y = np.asanyarray(train[['CO2EMISSIONS']])  # Var dependente
regr.fit(x, y)  # Fazendo a regressão com os X e Y determinados
# Coeficientes da regressão realizada.
print(f'Os coeficientes foram: {regr.coef_}')

# Prediction (Calculo de Erros)
# Abaixo estamos encontro o Y chapéu, cujo é utilizado para calcular os erros entre o valor de teste
# e o valor real que utilizamos.
y_hat = regr.predict(
    test[['ENGINESIZE', 'CYLINDERS',  'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])
rss = np.mean((y_hat - y) ** 2)
print(f"Residual sum of squares (RSS): {rss}")
variance = regr.score(x, y)
print(f"O score de variancia foi: {variance}")
