import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()

import matplotlib as mpl

import sklearn
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# importujemy zbiór danych
iris = datasets.load_iris()

X = iris.data  # macierz obiektów
y = iris.target  # wektor ich poprawnej klasyfikacji
target_names = iris.target_names  # nazwa klasy obiektu

# Uruchamiamy algorytm PCA dla 2 komponentów
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# Procent wariancji wyjaśnianej przez metodę dla każdego komponentu
print('Współczynnik wyjaśnianej wariancji dla 2 komponentów: %s'
      % str(pca.explained_variance_ratio_))

# Tworzymy wykres
plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend()
plt.show()
