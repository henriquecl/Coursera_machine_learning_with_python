import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
# %matplotlib inline

# Lendo os dados (Já tinha aprendido em cursos anteriores)
df = pd.read_csv('FuelConsumptionCo2.csv')
df.describe()  # Da uma descrição dos dados contidos, como média, mediana, DP, min, max
# O comando abaixo coloca em cdf as colunas de df selecionadas.
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(cdf.head(9))  # Imprime as 9 primeiras linhas


"""
Plotando um histograma
"""

viz = cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
viz.hist()
plt.show()

"""
Plotando os Dados em um gráfico X-Y (linear)
plt.scatter: É  uma função que plota dados da seguinte forma, (X,Y)
plt.xlabel: Nomeia o eixo x
plt.ylabel: Nomeia o eixo y
"""
plt.scatter(cdf['FUELCONSUMPTION_COMB'], cdf['CO2EMISSIONS'])
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('CO2_Emission')
plt.show()

# Plotando outro gráfico (Dado)

plt.scatter(cdf['ENGINESIZE'], cdf['CO2EMISSIONS'],  color='black')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Practice: plot cylinder vs Emission, to see how linear is their relation

plt.scatter(cdf['CYLINDERS'], cdf['CO2EMISSIONS'], color='green')
plt.xlabel("teste")
plt.xlabel('CO2 Emissions')
plt.show()
