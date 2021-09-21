# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:18:34 2021

@author: Àlex Correa
"""

from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Carreguem dataset d'exemple
dataset = load_dataset('HRDataset_v14_.csv')
data = dataset.values

x = data[:, :311]
y = data[:, 35]

print("Dimensionalitat de la BBDD:", dataset.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)

print("Per comptar el nombre de valors no existents:")
print(dataset.isnull().sum())

print(dataset)

# mostrem el "salary"
plt.figure()

ax = plt.scatter(x[:,9], y)

#HISTOGRAMA
plt.figure()
plt.title("Histograma de l'atribut 9")
plt.xlabel("Salary")
plt.ylabel("N. Persones")
hist = plt.hist(x[:,9], bins=11, range=[np.min(x[:,9]), np.max(x[:,9])], histtype="bar", rwidth=0.8)

#CORRELACIÓ
import seaborn as sns

# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
correlacio = dataset.corr()

plt.figure()

ax = sns.heatmap(correlacio, annot=True, linewidths=.5)

# Mirem la relació entre atributs utilitzant la funció pairplot
relacio = sns.pairplot(dataset)

