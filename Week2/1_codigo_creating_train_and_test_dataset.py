import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
from sklearn import linear_model

df = pd.read_csv('FuelConsumptionCo2.csv')
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
"""
Como sabemos, para realizar o treino e o de teste do nosso dataset, deveremos separa-lo em 2 grupos
óbviamente o de treino e o de teste, cujos não compartilham dados em comum. Devemos treinar apenas
no dataset de treino, e testar apenas no dataset de teste. Mas por que isso? É feito dessa forma 
pois, como os dados não são compartilhados entre os datasets a acurácia do que é chamado de 
"evaluation on out-of-sample"(em outras palavras, a chance de um dado externo ser confiável aumenta)
tornando assim o nosso programa mais confiável e mais próximo do mundo real.
"""
# Separando os datasets, 80% para treino e 20% para teste.
msk = np.random.rand(len(df)) < 0.8   # Seleciona linhas aleatórias.
train = cdf[msk]
test = cdf[~msk]    # O '~' Retorna o complemento de msk q não foi utilizado.

# Train data distribution.
plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'], color='blue')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emission')
plt.show()
"""
O que estamos fazendo abaixo?
    Relembrando da aula teórica, nós faziamos os cálculos por uma fórmula matematica apresentada 
pelo instrutor. Nas linhas abaixo, os coeficientes são calculados através do sklearn. Ou seja,
é encontrado o coeficiente angular e o coeficiente linear.
"""
# Fazendo a regressão linear com sklearn
regr = linear_model.LinearRegression()
# Transforma a coluna em um array 
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print(f'Coefficients:{regr.coef_} ')
print(f'Intercept: {regr.intercept_}')

# Plot outputs
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# Na linha abaixo estamos plotando o plt.scattter mais a função ax + b
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

""" 
Evaluation (Analise de erros)

There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our 
model based on the test set:
    Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the
 metrics to understand since it’s just average error.
    Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more
popular than Mean absolute error because the focus is geared more towards large errors. 
This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
    Root Mean Squared Error (RMSE): This is the square root of the Mean Square Error.
    R-squared is not error, but is a popular metric for accuracy of your model. It represents how 
close the data are to the fitted regression line. The higher the R-squared, the better the model 
fits your data. Best possible score is 1.0 and it can be negative (because the model can be 
arbitrarily worse).
"""
# Utilizando o MSE
from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
