import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Importing de dataframe
df = pd.read_csv('FuelConsumptionCo2.csv')

# Criando o dataset de treino e teste
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

# Separando as variáveis no dataset se treino e teste
# Treino
x_train = np.asanyarray(train[['ENGINESIZE']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
# Teste
x_test = np.asanyarray(test[['ENGINESIZE']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])

# Fazendo a regressão polinomial de maneira cúbica
poly = PolynomialFeatures(degree=3)
x_train_poly = poly.fit_transform(x_train)
print(x_train_poly)

# Transformando a regressão polinomial em linear multipla
clf = linear_model.LinearRegression()
y_train_ = clf.fit(x_train_poly, y_train)
print(f'Os coeficientes foram: {clf.coef_}\nE os interceptos: {clf.intercept_}')

# Com os parametros da curva vamos plotá-la de maneira correta
plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'], color='blue')
XX = np.arange(0, 10, 0.1)
YY = clf.intercept_[0] + clf.coef_[0][1]*XX + clf.coef_[0][2]*np.power(XX, 2)+clf.coef_[0][3]*np.power(XX,3)
plt.plot(XX, YY, color='red')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.show()

# Evaluation

x_test_poly = poly.fit_transform(x_test)
y_test_hat = clf.predict(x_test_poly)
MAE = np.mean(np.absolute(y_test_hat - y_test))
MSE = np.mean((y_test_hat - y_test)**2)
R2 = r2_score(y_test_hat, y_test)
print(f'O mean absolute erros foi: {MAE}\nO residual sum of squares: {MSE}\nO R2-score foi: {R2}')
