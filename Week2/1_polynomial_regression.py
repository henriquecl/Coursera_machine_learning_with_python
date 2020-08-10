import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# Lendo o dataframe
df = pd.read_csv('FuelConsumptionCo2.csv')

# Selecionando quais dados iremos fazer a regressão
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Plotando os dados para saber que tipo de regressão iremos fazer.
plt.scatter(cdf['ENGINESIZE'], cdf['CO2EMISSIONS'], color='blue')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.show()

# Criando o train and test dataset
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

# Separando teste e treino
x_train = np.asanyarray(train[['ENGINESIZE']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])

x_test = np.asanyarray(test[['ENGINESIZE']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])
# Fazendo a regressão polinomial
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
"""
A função fit_transform pega todos os valores dos nossos dados e o transforma em uma matriz cujo cada
coluna o valor é elevado a um expoente definido pelo degree=2. Ou seja, a coluna 0 é o dado ^ 0,
a coluna 1 é o dado ^1 , e a coluna 2 é o dado ^ 2.
"""
print(x_train_poly)

# Como a regressão polinomial é um caso particular da regressão linear múltipla, podemos trata-la
# como uma regressão linear multipla. LOGO, FAZEMOS:
clf = linear_model.LinearRegression()
y_train_ = clf.fit(x_train_poly, y_train)
print(f'Os coeficientes foram: {clf.coef_}')
print(f'Os interceptos foram: {clf.intercept_}')

# Tendo os parametros da nossa curva podemos plota-la corretamente.
plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'], color='blue')
XX = np.arange(0.0, 10.0, 0.1)  # O passo com que cada ponto será plotado no gráfico.
YY = clf.intercept_[0] + clf.coef_[0][1]*XX + clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, YY, color='red')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.show()

# Evaluation
from sklearn.metrics import r2_score
x_test_poly = poly.fit_transform(x_test)
y_test_hat = clf.predict(x_test_poly)
MAE = np.mean(np.absolute(y_test_hat - y_test))
MSE = np.mean((y_test_hat - y_test)**2)
R2 = r2_score(y_test_hat, y_test)
print(f'O mean absolute erros foi: {MAE}\nO residual sum of squares: {MSE}\nO R2-score foi: {R2}')
